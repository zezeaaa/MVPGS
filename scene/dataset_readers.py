#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from dataclasses import dataclass
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

from mvs_modules.mvs_estimator import MvsEstimator

@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

    K: np.array = None
    bounds: np.array = None
    mvs_depth: np.array = None
    mvs_mask: np.array = None
    fg_mask: np.array = None # for DTU evaluation
    mono_depth: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, load_fg_mask=False, dtu_mask_path=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0., intr.params[1]],
                [0., focal_length_x, intr.params[2]],
                [0., 0., 1.]
            ], dtype=np.float32)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0., intr.params[2]],
                [0., focal_length_y, intr.params[3]],
                [0., 0., 1.]
            ], dtype=np.float32)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # read scene bounds
        pose_file_path = os.path.join(os.path.dirname(images_folder), 'poses_bounds.npy')
        poses_arr = np.load(pose_file_path)
        bds = poses_arr[extr.id-1, -2:]     

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # read dtu foregroud mask
        dtu_mask = None
        if load_fg_mask:
            scene_name = image_path.split('/')[-3]
            idx = int(image_name.split('_')[1]) - 1
            dtu_mask_file = os.path.join(dtu_mask_path, scene_name, f'{idx:03d}.png')
            if not os.path.exists(dtu_mask_file):
                dtu_mask_file = os.path.join(dtu_mask_path, scene_name, f'mask/{idx:03d}.png')
            if os.path.exists(dtu_mask_file):
                dtu_mask = np.array(Image.open(dtu_mask_file), dtype=np.float32)[:, :, :3] / 255.
                dtu_mask = (dtu_mask == 1)[:,:,0]
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, bounds=bds, fg_mask=dtu_mask, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, dataset='LLFF', input_n=3, mvs_config_path=None, use_colmap_init=False, pc_downsample=0.1, stage='train', dtu_mask_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    load_fg_mask = True if dataset=='DTU' else False
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), 
                                           load_fg_mask=load_fg_mask, dtu_mask_path=dtu_mask_path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    print('Dataset: ', dataset)
    # dataset split
    if eval:
        if dataset == 'DTU':
            print('Eval DTU Dataset!!!')
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
            train_cam_infos = [cam_infos[i] for i in train_idx[:input_n]]
            test_cam_infos = [cam_infos[i] for i in test_idx]
        elif dataset == 'LLFF':
            print('Eval LLFF Dataset!!!')
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            if input_n >= 1:
                idx_sub = np.linspace(0, len(train_cam_infos) - 1, input_n)
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        elif dataset == 'NVSRGBD':
            print('Eval NVSRGBD Dataset!!!')
            train_cam_infos = []
            test_cam_infos = []
            for cam_info in cam_infos:
                if 'train' in cam_info.image_name:
                    train_cam_infos.append(cam_info)
                    print('Train: ', cam_info.image_name)
                else:
                    test_cam_infos.append(cam_info)
                    print('Test: ', cam_info.image_name)
        elif dataset == 'Tank':
            print('Eval Tank Dataset!!!')
            cam_infos = cam_infos[:50] # we use the first 50 frames of each scene for our experiments
            print('Total cams: ', len(cam_infos))
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            if input_n >= 1:
                idx_sub = np.linspace(0, len(train_cam_infos) - 1, input_n)
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        else:
            raise NotImplementedError
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    if not use_colmap_init and stage == 'train':
        # get Mono depth
        print('Predicting Mono depth...')
        imgs = [np.asarray(cam.image.copy().convert('RGB')) for cam in train_cam_infos]
        mono_depths = get_mono_depth(imgs)
        for i, cam in enumerate(train_cam_infos):
            cam.mono_depth = mono_depths[i]

        # get MVS depth, initial GS positions
        print('Predicting MVS depth...')
        mvs_estimator = MvsEstimator(mvs_config_path)
        vertices, mvs_depths, masks = mvs_estimator.get_mvs_pts(train_cam_infos)
        torch.cuda.empty_cache()
        for i, cam in enumerate(train_cam_infos):
            cam.mvs_depth = mvs_depths[i]
            cam.mvs_mask = masks[i]

        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.zeros_like(positions)

        print('Points num: ', len(positions))
        # random down sample
        if pc_downsample < 1.0:
            random_idx = np.random.choice(positions.shape[0], int(positions.shape[0] * pc_downsample), replace=False)
            positions = positions[random_idx]
            colors = colors[random_idx]
            normals = normals[random_idx]

        pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
        print(f"Initial points num: {positions.shape[0]}")
        ply_path = None
        del mvs_estimator

    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def get_mono_depth(imgs):
    # get Mono depth (from https://github.com/Wanggcong/SparseNeRF/blob/main/get_depth_map_for_llff_dtu.py)
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    H, W = imgs[0].shape[0:2]
    imgs = torch.concat([transform(img).to(device) for img in imgs])

    with torch.no_grad():
        prediction = midas(imgs)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}