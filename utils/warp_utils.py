# Shree KRISHNAya Namaha
# Differentiable warper implemented in PyTorch. Warping is done on batches.
# Tested on PyTorch 1.8.1
# Author: Nagabhushan S N
# Last Modified: 27/09/2021

import datetime
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional

import numpy
import skimage.io
import torch
import torch.nn.functional as F

class Warper:
    def __init__(self, resolution: tuple = None, device: str = 'gpu0'):
        self.resolution = resolution
        self.device = self.get_device(device)
        return

    def forward_warp(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                     transformation1: torch.Tensor, transformation2: torch.Tensor, intrinsic1: torch.Tensor, 
                     intrinsic2: Optional[torch.Tensor]) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                        bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                        method accordingly.
        :param mask1: (b, 1, h, w) - 1 for known, 0 for unknown. Optional
        :param depth1: (b, 1, h, w)
        :param transformation1: (b, 4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (b, 4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (b, 3, 3) camera intrinsic matrix
        :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        assert frame1.shape == (b, 3, h, w)
        assert mask1.shape == (b, 1, h, w)
        assert depth1.shape == (b, 1, h, w)
        assert transformation1.shape == (b, 4, 4)
        assert transformation2.shape == (b, 4, 4)
        assert intrinsic1.shape == (b, 3, 3)
        assert intrinsic2.shape == (b, 3, 3)

        frame1 = frame1.to(self.device)
        mask1 = mask1.to(self.device)
        depth1 = depth1.to(self.device)
        transformation1 = transformation1.to(self.device)
        transformation2 = transformation2.to(self.device)
        intrinsic1 = intrinsic1.to(self.device)
        intrinsic2 = intrinsic2.to(self.device)

        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, intrinsic1,
                                                        intrinsic2)
        trans_coordinates = trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, :, 2, 0].unsqueeze(1)

        grid = self.create_grid(b, h, w).to(trans_coordinates)
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid

        warped_frame2, mask2 = self.bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
        warped_depth2 = self.bilinear_splatting(trans_depth1, mask1, trans_depth1, flow12, None,
                                                is_image=False)[0]
        return warped_frame2, mask2, warped_depth2, flow12

    def compute_transformed_points(self, depth1: torch.Tensor, transformation1: torch.Tensor, transformation2: torch.Tensor,
                                   intrinsic1: torch.Tensor, intrinsic2: Optional[torch.Tensor]):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        b, _, h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        transformation = torch.bmm(transformation2, torch.linalg.inv(transformation1))  # (b, 4, 4)

        x1d = torch.arange(0, w)[None]
        y1d = torch.arange(0, h)[:, None]
        x2d = x1d.repeat([h, 1]).to(depth1)  # (h, w)
        y2d = y1d.repeat([1, w]).to(depth1)  # (h, w)
        ones_2d = torch.ones(size=(h, w)).to(depth1)  # (h, w)
        ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])  # (b, h, w, 1, 1)
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (b, h, w, 3, 1)
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
        return trans_norm_points

    def bilinear_splatting(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                           flow12: torch.Tensor, flow12_mask: Optional[torch.Tensor], is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = torch.moveaxis(prox_weight_nw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(frame1)
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(frame1)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_nw), accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_sw), accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_ne), accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_se), accumulate=True)

        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                  weight_nw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                  weight_sw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                  weight_ne, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                  weight_se, accumulate=True)

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
        mask2 = mask.to(frame1)

        if is_image:
            assert warped_frame2.min() >= -1.1  # Allow for rounding errors
            assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=0, max=1)
        return warped_frame2, mask2

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def get_device(device: str):
        """
        Returns torch device object
        :param device: cpu/gpu0/gpu1
        :return:
        """
        if device == 'cpu':
            device = torch.device('cpu')
        elif device.startswith('gpu') and torch.cuda.is_available():
            gpu_num = int(device[3:])
            device = torch.device(f'cuda:{gpu_num}')
        else:
            device = torch.device('cpu')
        return device
