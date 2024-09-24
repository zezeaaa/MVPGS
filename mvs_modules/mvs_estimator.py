import os
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from .utils import read_json
from .mvsformer.mvsformer_model import TwinMVSNet, DINOMVSNet
from .fusion import *

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting
class MvsEstimator:
    def __init__(self, config_path, scale_bds=[1.0, 1.0]):
        self.config = read_json(config_path)
        self.scale_bds = scale_bds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.create_model(self.config)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_mvs_pts(self, input_cams, save_path=None):
        all_ori_imgs = []
        mvs_depths = []
        photometric_confidences = []
        cam_mats = []
        # estimate mvs depth for each view
        for ref_view in tqdm(range(len(input_cams)), desc='MVS Estimation'):
            input_cams_copy = input_cams.copy()
            ref_cam = input_cams_copy.pop(ref_view) # ref view
            all_cams = [ref_cam] + input_cams_copy # ref view + src view

            # prepare MVS inputs
            imgs, ori_imgs, proj_matrices, ref_depth_bins = self.prepare_mvs_input(all_cams, bds_scale_factor=self.scale_bds)
            proj_matrices_ms = self.get_ms_proj_mats(proj_matrices)

            with torch.no_grad():
                outputs = self.model.forward(imgs, proj_matrices_ms, ref_depth_bins, tmp=self.config['tmp'])

            depth_est = outputs["refined_depth"]
            photometric_confidence = outputs["photometric_confidence"]

            mvs_depths.append(depth_est)
            photometric_confidences.append(photometric_confidence)
            cam_mats.append(proj_matrices[:,0,:,:,:])
            all_ori_imgs.append(ori_imgs[0])
        
        # merge MVS depth into point cloud
        points, masks = self.merge_mvs_depths(all_ori_imgs, mvs_depths, photometric_confidences, cam_mats, prob_threshold=self.config['prob_threshold'])
        mvs_depths = [mvs_depths[i].squeeze().cpu().numpy() for i in range(len(mvs_depths))]

        if save_path is not None:
            vis_path = os.path.join(save_path, 'vis')
            os.makedirs(vis_path, exist_ok=True)

            for v in range(len(input_cams)):
                file_name = input_cams[v].image_name
                depth_est = mvs_depths[v]
                vis_depth = colormap(depth_est)
                plt.imsave(os.path.join(vis_path, file_name + '.png'), vis_depth.transpose((1,2,0))) # (H, W, 3)
                plt.imsave(os.path.join(vis_path, file_name+'_mask.png'), masks[v], cmap='gray')
            
            el = PlyElement.describe(points, 'vertex')
            PlyData([el]).write(os.path.join(vis_path, 'mvs_input.ply'))
        
        return points, mvs_depths, masks


    def merge_mvs_depths(self, imgs, depths, confidences, input_cams, prob_threshold):
        views = {}
        masks = []
        for ref_view in range(len(input_cams)):
            ref_depth, ref_cam = depths[ref_view].unsqueeze(0), input_cams[ref_view]
            ref_conf = confidences[ref_view].unsqueeze(0)
            ref_prob_mask = ref_conf > prob_threshold[0]

            src_depths = torch.stack(depths[:ref_view] + depths[ref_view+1:], dim=0).unsqueeze(0)
            src_cams = torch.stack(input_cams[:ref_view] + input_cams[ref_view+1:], dim=0).unsqueeze(0)
            src_confs = torch.stack(confidences[:ref_view] + confidences[ref_view+1:], dim=0).unsqueeze(0)
            src_prob_mask = src_confs > prob_threshold[0]
            src_depths *= src_prob_mask.float()

            reproj_xyd, in_range = get_reproj(ref_depth, src_depths, ref_cam, src_cams)
            vis_masks, vis_mask = vis_filter(ref_depth, reproj_xyd, in_range, self.config['thres_disp'], 0.01, self.config['thres_view'])
            ref_depth_ave = ave_fusion(ref_depth, reproj_xyd, vis_masks)

            mask = bin_op_reduce([ref_prob_mask, vis_mask], torch.min)

            idx_img = get_pixel_grids(*ref_depth_ave.size()[-2:]).unsqueeze(0)
            idx_cam = idx_img2cam(idx_img, ref_depth_ave, ref_cam)
            points = idx_cam2world(idx_cam, ref_cam)[..., :3, 0].permute(0, 3, 1, 2)

            points_np = points.cpu().data.numpy()
            # prob_mask_np = ref_prob_mask.cpu().data.numpy().astype(np.bool_)
            # vis_mask_np = vis_mask.cpu().data.numpy().astype(np.bool_)
            mask_np = mask.cpu().data.numpy().astype(np.bool_)
            masks.append(mask_np.squeeze())

            ref_img = imgs[ref_view].transpose(2, 0, 1)[np.newaxis, ...]
            for i in range(points_np.shape[0]):
                # print(np.sum(np.isnan(points_np[i])))
                p_f_list = [points_np[i, k][mask_np[i, 0]] for k in range(3)]
                p_f = np.stack(p_f_list, -1)
                c_f_list = [ref_img[i, k][mask_np[i, 0]] for k in range(3)]
                c_f = np.stack(c_f_list, -1) * 255
                # d_f_list = [dir_vecs[i, k][mask_np[i, 0]] for k in range(3)]
                # d_f = np.stack(d_f_list, -1)
                views[ref_view] = (p_f, c_f.astype(np.uint8))

        p_all, c_all = [np.concatenate([v[k] for key, v in views.items()], axis=0) for k in range(2)]

        vertexs = np.array([tuple(v) for v in p_all], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_colors = np.array([tuple(v) for v in c_all], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
        for prop in vertexs.dtype.names:
            vertex_all[prop] = vertexs[prop]
        for prop in vertex_colors.dtype.names:
            vertex_all[prop] = vertex_colors[prop]

        return vertex_all, masks

    def prepare_mvs_input(self, input_cams, num_depths=192, bds_scale_factor=[1.0, 1.0]):
        imgs = []
        ori_imgs = []
        proj_matrices = []
        for cam in input_cams:
            img = cam.image.copy().convert('RGB')
            ori_img, intrinsic = self.scale_mvs_input(np.array(img), cam.K, self.config['max_w'], self.config['max_h'])
            img = self.transforms(Image.fromarray(ori_img))
            extrinsic = getWorld2View(cam.R, cam.T)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsic
            proj_mat[1, :3, :3] = intrinsic

            imgs.append(img)
            ori_imgs.append(ori_img / 255.)
            proj_matrices.append(proj_mat)

        imgs = torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)
        proj_matrices = torch.tensor(np.stack(proj_matrices, axis=0)).unsqueeze(0).to(self.device)
        # read depth bounds, better performance can be achieved by adjust depth bounds. Here we use the default setting.
        depth_min, depth_max = input_cams[0].bounds[0]*bds_scale_factor[0], input_cams[0].bounds[1]*bds_scale_factor[1]
        ref_depth_bins = torch.linspace(depth_min, depth_max, num_depths, dtype=torch.float32).unsqueeze(0).to(self.device)

        return imgs, ori_imgs, proj_matrices, ref_depth_bins
    
    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=64):
        new_intrinsics = intrinsics.copy()
        h, w = img.shape[:2]
        new_h, new_w = max_h, max_w

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        new_intrinsics[0, :] *= scale_w
        new_intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, new_intrinsics

    def get_ms_proj_mats(self, proj_matrices):
        # multi-stage
        stage0_pjmats = proj_matrices.clone()
        stage0_pjmats[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 0.125
        stage1_pjmats = proj_matrices.clone()
        stage1_pjmats[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 0.25
        stage2_pjmats = proj_matrices.clone()
        stage2_pjmats[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 0.5
        stage3_pjmats = proj_matrices.clone()
        stage3_pjmats[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]
        stage4_pjmats = proj_matrices.clone()
        stage4_pjmats[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]

        proj_matrices_ms = {
            "stage1": stage0_pjmats,
            "stage2": stage1_pjmats,
            "stage3": stage2_pjmats,
            "stage4": stage3_pjmats,
            "stage5": stage4_pjmats
        }

        return proj_matrices_ms

    def create_model(self, config):
        # model
        # build models architecture, then print to console
        if config['arch']['args']['vit_args'].get('twin', False):
            model = TwinMVSNet(config['arch']['args'])
        else:
            model = DINOMVSNet(config['arch']['args'])

        print('Loading checkpoint: {} ...'.format(self.config['model_path']))
        checkpoint = torch.load(str(self.config['model_path']))
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, val in state_dict.items():
            new_state_dict[key.replace('module.', '')] = val
        model.load_state_dict(new_state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis