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
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from simple_lama_inpainting import SimpleLama
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import getWorld2View, focal2fov

from utils.warp_utils import Warper
from utils.pose_utils import generate_pseudo_poses_llff, generate_random_poses_dtu

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], stage='render'):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.args = args
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.stage = stage

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.virtual_cameras = {} # unseen view virtual cams
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, 
                                                          dataset=args.dataset, input_n=args.input_views, 
                                                          mvs_config_path=args.mvs_config, pc_downsample=args.init_pc_downsample, 
                                                          stage=self.stage, dtu_mask_path=args.dtu_mask_path)
            
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if scene_info.ply_path is not None:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            # get unseen cameras
            if self.stage == 'train':
                print("Generating Virtual Cameras, num: ", args.total_virtual_num)
                if args.dataset == 'DTU':
                    self.virtual_cameras[resolution_scale] = self.generateVirtualCams(self.train_cameras[resolution_scale], v_num=args.total_virtual_num, use_mask=True)
                else:
                    self.virtual_cameras[resolution_scale] = self.generateVirtualCams(self.train_cameras[resolution_scale], v_num=args.total_virtual_num, inpaint=True)
                torch.cuda.empty_cache()


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getVirtualCameras(self, scale=1.0):
        return self.virtual_cameras[scale]
    
    def getAllCameras(self, scale=1.0):
        return self.train_cameras[scale] + self.test_cameras[scale]

    def generateVirtualCams(self, input_cams, v_num=120, batch_size=24, use_mask=False, inpaint=False):
        print('Train view num: ', len(input_cams))
        input_imgs, mvs_depths, masks, input_extrs, input_intrs, \
        target_extrs, target_intrs = self.prepare_data(input_cams, v_num)

        # split into batches
        input_batches = create_batches(input_imgs, mvs_depths, masks, input_extrs, input_intrs, batch_size=batch_size)
        target_batches = create_batches(target_intrs, target_extrs, batch_size=batch_size)

        warper = Warper()
        warped_frames, valid_masks, warped_depths = [], [], []
        with torch.no_grad():
            for (input_imgs_batch, mvs_depths_batch, masks_batch, input_extrs_batch, input_intrs_batch), (target_intrs_batch, target_extrs_batch) \
                in tqdm(zip(input_batches, target_batches), desc="Unseen view prior", unit="batch", total=int(v_num / batch_size)):
                torch.cuda.empty_cache()
                masks_batch = None if not use_mask else masks_batch
                # get priors for unseen views by forward warping
                warped_frame, valid_mask, warped_depth, _ = warper.forward_warp(input_imgs_batch, masks_batch, mvs_depths_batch, input_extrs_batch, 
                                                                                target_extrs_batch, input_intrs_batch, target_intrs_batch)
                warped_frames.append(warped_frame.cpu())
                valid_masks.append(valid_mask.cpu())
                warped_depths.append(warped_depth.cpu())
                

        warped_depths = torch.cat(warped_depths, dim=0)
        valid_masks = torch.cat(valid_masks, dim=0)
        warped_frames = torch.cat(warped_frames, dim=0)

        virtual_cams = []
        if inpaint:
            simple_lama = SimpleLama() # use Lama for inpainting if needed
        for i in range(v_num):
            id = len(input_cams) + i
            R, T = target_extrs[i, :3, :3].cpu().numpy().transpose(), target_extrs[i, :3, 3].cpu().numpy()
            focal_length_x, focal_length_y = target_intrs[i][0,0], target_intrs[i][1,1]
            H, W = warped_frames.shape[2:4]
            FovY = focal2fov(focal_length_y, H)
            FovX = focal2fov(focal_length_x, W)
            warped_img = warped_frames[i]
            mask = valid_masks[i].squeeze().to(torch.bool).detach().cpu().numpy()
            if inpaint:# inpaint
                warped_img = warped_img.permute(1,2,0).cpu().numpy()
                warped_img = torch.from_numpy(np.array(simple_lama(warped_img*255, (~mask).astype(np.uint8)*255))[:H, :W]).permute(2,0,1)/255.
            
            virtual_cam = Camera(colmap_id=None, R=R, T=T, 
                                FoVx=FovX, FoVy=FovY, 
                                image=warped_img, gt_alpha_mask=None,
                                image_name='virtual_'+str(i), uid=id, data_device='cpu',
                                K=target_intrs[i].cpu().numpy(), mvs_depth=warped_depths[i].cpu().numpy(), mask=valid_masks[i].cpu().numpy(), is_virtual=True)
            virtual_cams.append(virtual_cam)

        return virtual_cams


    def prepare_data(self, input_cams, v_num):
        # choose one source view randomly
        ids = np.random.choice(len(input_cams), v_num, replace=True)
            
        input_imgs = torch.stack([input_cams[id].original_image for id in ids])
        mvs_depths = torch.from_numpy(np.stack([input_cams[id].mvs_depth for id in ids])).unsqueeze(1)
        masks = torch.from_numpy(np.stack([input_cams[id].mask for id in ids])).unsqueeze(1)
        input_extrs = torch.from_numpy(np.stack([getWorld2View(input_cams[id].R, input_cams[id].T) for id in ids]))
        input_intrs = torch.from_numpy(np.stack([input_cams[id].K for id in ids]))

        # generate random poses for unseen views
        if self.args.dataset == 'LLFF' or self.args.dataset == 'Tank' or self.args.dataset == 'NVSRGBD':
            bds = np.stack([cam.bds for cam in input_cams])
            target_poses = torch.from_numpy(generate_pseudo_poses_llff(input_extrs, bds, n_poses=v_num))
        elif self.args.dataset == 'DTU':
            target_poses = torch.from_numpy(generate_random_poses_dtu(input_extrs, n_poses=v_num))
        else:
            raise ValueError("Unknown dataset: {}".format(self.args.dataset))

        target_extrs = torch.from_numpy(np.stack([np.linalg.inv(target_poses[i]) for i in range(target_poses.shape[0])]))
        target_intrs = torch.from_numpy(np.stack([input_cams[0].K] * len(ids)))  # same intrinsics

        return input_imgs.cpu(), mvs_depths.cpu(), masks.cpu(), input_extrs, input_intrs, target_extrs, target_intrs

def create_batches(*tensors: torch.Tensor, batch_size: int):
    return list(zip(*[torch.split(tensor, batch_size) for tensor in tensors]))