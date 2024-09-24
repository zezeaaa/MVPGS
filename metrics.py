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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as tf
# from utils.loss_utils import ssim
from skimage.metrics import structural_similarity

# from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np

# use NeRF's evaluation tools
def ssim(x, y):
    ## ---- fix the SSIM issue default in regnerf's code ---- ##
    x = x.cpu().numpy().squeeze().transpose(1,2,0)
    y = y.cpu().numpy().squeeze().transpose(1,2,0)
    return structural_similarity(x, y, channel_axis=-1, data_range=1.0)
    # ------------------------------------------------------ ##
from lpips import LPIPS

lpips_vgg = LPIPS(net="vgg")
def lpips_fn(x, y):
    score = lpips_vgg(x.cpu(), y.cpu())
    return score.item()

def readImages(renders_dir, gt_dir, mask_dir=None):
    renders = []
    gts = []
    image_names = []
    masks = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        if os.path.exists(mask_dir / fname):
            mask = np.array(Image.open(mask_dir / fname)) // 255
            masks.append(tf.to_tensor(mask==1).squeeze().cuda())
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names, masks

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                mask_dir = method_dir / "mask"
                renders, gts, image_names, masks = readImages(renders_dir, gt_dir, mask_dir)

                ssims = []
                psnrs = []
                lpipss = []
                masked_psnrs = []
                masked_ssims = []
                masked_lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]))
                    if len(masks) > 0:
                        masked_psnrs.append(psnr(renders[idx][:, :, masks[idx]], gts[idx][:, :, masks[idx]]))
                        rgb_fg = gts[idx] * masks[idx] + (~masks[idx])
                        rgb_hat_fg = renders[idx] * masks[idx] + (~masks[idx])
                        os.makedirs(os.path.join(method_dir, 'masked_rendering'), exist_ok=True)
                        torchvision.utils.save_image(rgb_hat_fg, os.path.join(method_dir, 'masked_rendering', image_names[idx]))
                        masked_ssims.append(ssim(rgb_fg, rgb_hat_fg))
                        masked_lpipss.append(lpips_fn(rgb_fg, rgb_hat_fg))

                    else:
                        masked_psnrs.append(0.0)
                        masked_ssims.append(0.0)
                        masked_lpipss.append(0.0)

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  Masked PSNR: {:>12.7f}".format(torch.tensor(masked_psnrs).mean(), ".5"))
                print("  Masked SSIM: {:>12.7f}".format(torch.tensor(masked_ssims).mean(), ".5"))
                print("  Masked LPIPS: {:>12.7f}".format(torch.tensor(masked_lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "Masked SSIM": torch.tensor(masked_ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "Masked PSNR": torch.tensor(masked_psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "Masked LPIPS": torch.tensor(masked_lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "Masked SSIM": {name: masked_ssim for masked_ssim, name in zip(torch.tensor(masked_ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "Masked PSNR": {name: masked_psnr for masked_psnr, name in zip(torch.tensor(masked_psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "Masked LPIPS": {name: masked_lpips for masked_lpips, name in zip(torch.tensor(masked_lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
