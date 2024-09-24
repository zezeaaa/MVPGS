import numpy as np
from typing import Tuple

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def focus_pt_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def poses_avg(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world

def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def generate_random_poses_dtu(extrinsics, n_poses=120, r_scale=4.0):
    poses = np.stack([np.linalg.inv(extrinsics[i]) for i in range(extrinsics.shape[0])])
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0) * r_scale
    radii = np.concatenate([radii, [1.]])
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    z_axis = focus_pt_fn(poses)
    random_poses = []
    for _ in range(n_poses):
        random_pose = np.eye(4, dtype=np.float32)
        t = radii * np.concatenate([
            2 * 1.0 * (np.random.rand(3) - 0.5), [1,]])
        position = cam2world @ t
        z_axis_i = z_axis + np.random.randn(*z_axis.shape) * 0.125
        random_pose[:3, :4] = viewmatrix(z_axis_i, up, position, True)
        random_poses.append(random_pose)
    random_poses = np.stack(random_poses, axis=0)
    return random_poses

def generate_pseudo_poses_llff(extrinsics, bounds, n_poses, r_scale=2.0):
    """Generates random poses."""
    poses = np.stack([np.linalg.inv(extrinsics[i]) for i in range(extrinsics.shape[0])])

    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0) * r_scale
    radii = np.concatenate([radii, [1.]])

    # Generate random poses.
    random_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for _ in range(n_poses):
        random_pose = np.eye(4, dtype=np.float32)
        t = radii * np.concatenate([2 * np.random.rand(3) - 1., [1,]])
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.]
        z_axis = position - lookat
        random_pose[:3, :4] = viewmatrix(z_axis, up, position)
        random_poses.append(random_pose)

    return np.stack(random_poses, axis=0)
