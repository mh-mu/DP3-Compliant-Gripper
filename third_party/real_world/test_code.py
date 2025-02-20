# bash scripts/metaworld/gen_demonstration_expert.sh reach 5
import argparse
import os
import zarr
import numpy as np
from diffusion_policy_3d.env.real_world import RealWorldEnv
from termcolor import cprint
import copy
import imageio
import cv2
from metaworld.policies import *
# import faulthandler
# faulthandler.enable()

import pyspacemouse
from scipy.spatial.transform import Rotation
from klampt.math import so3, se3
from klampt.model import trajectory
from diffusion_policy_3d.env.real_world import CONSTANTS
import keyboard
from icecream import ic 

from tqdm import tqdm
import torch

# # Example usage of the klampt.model.trajectory library

# # Create a trajectory with a list of milestones (waypoints)
# milestones = [
#     [0, 0, 0],
#     [1, 1, 1],
#     [2, 2, 2],
#     [3, 3, 3]
# ]

# # Create a Trajectory object
# traj = trajectory.Trajectory(milestones=milestones)

# # Print the trajectory
# print("Trajectory milestones:", traj.milestones)

# # Evaluate the trajectory at a specific time
# time = 1.5
# point = traj.eval(time)
# print(f"Point at time {time}:", point)

# # Save the trajectory to a file
# traj.save("trajectory.path")

# # Load the trajectory from a file
# loaded_traj = trajectory.Trajectory()
# loaded_traj.load("trajectory.path")
# print("Loaded trajectory milestones:", loaded_traj.milestones)

# # Create two rotation matrices using scipy.spatial.transform.Rotation
# rotation1 = Rotation.from_euler('xyz', [13, 52, 53], degrees=True).as_matrix()
# rotation2 = Rotation.from_euler('xyz', [7, 43, 6], degrees=True).as_matrix()

# # Flatten the rotation matrices to lists, column major form
# r1 = rotation1.T.flatten().tolist()
# r2 = rotation2.T.flatten().tolist()

# # Concatenate r1 and r2 into an array of shape (n, 9)
# rotations = np.array([r1, r2])
# ic(rotations)

# rotation_matrices = rotations.reshape(-1, 3, 3, order='F')
# ic(rotation_matrices)
# rotation_matrices = rotation_matrices.astype(np.float32)
# ic(rotation_matrices)
# rotation_diff_matrices = np.array([r2 @ np.linalg.inv(r1) for r1, r2 in tqdm(zip(rotation_matrices[:-1], rotation_matrices[1:]), total=len(rotation_matrices) - 1)], dtype=np.float32)
# ic(rotation_diff_matrices)

# reverse = np.linalg.inv(rotation_diff_matrices) @ rotation2
# ic(reverse)
# ic(rotation1)

# # Convert the rotation difference matrices to Euler angles
# rotation_diff_euler = np.array([Rotation.from_matrix(r).as_euler('xyz', degrees=True) for r in rotation_diff_matrices], dtype=np.float32)
# ic(rotation_diff_euler)

# ic(rotation_diff_matrices)
# rotation_diff_matrices = rotation_diff_matrices[:, :-1, :]  # Remove the last row of each 3 by 3 rotation matrix
# ic(rotation_diff_matrices)
# rotation_diff = rotation_diff_matrices.reshape(-1, 6)
# ic(rotation_diff)




# from klampt.model import trajectory

# # Create two random rotation matrices using scipy.spatial.transform.Rotation
# rotation1 = Rotation.random().as_matrix()
# rotation2 = Rotation.random().as_matrix()

# # Print the rotation matrices
# print("Rotation matrix 1:\n", rotation1)
# print("Rotation matrix 2:\n", rotation2)

# # Flatten the rotation matrices to lists, column major form
# r1 = rotation1.T.flatten().tolist()
# r2 = rotation2.T.flatten().tolist()

# # Print the flattened rotation matrices
# print("Flattened rotation matrix 1:", r1)
# print("Flattened rotation matrix 2:", r2)

# # Create two random translation vectors
# translation1 = np.random.rand(3)
# translation2 = np.random.rand(3)

# # Print the translation vectors
# print("Translation vector 1:", translation1)
# print("Translation vector 2:", translation2)

# # Combine the rotation list and the translation list into a single list of two, with the rotation list first
# combined1 = [r1, translation1.tolist()]
# combined2 = [r2, translation2.tolist()]

# # Print the combined lists
# print("Combined list 1:", combined1)
# print("Combined list 2:", combined2)

# traj = trajectory.SE3Trajectory(times=[0, 2], milestones=[combined1, combined2])
# ic(traj.eval(0))
# ic(traj.eval(1))
# ic(traj.eval(2))

# # ic(combined1)
# # rot_6d = np.array(combined1[0]).reshape(3, 3, order='F')[:2].flatten().tolist()
# # ic(rot_6d)


# import torch.nn.functional as F
# from einops import rearrange, reduce
# # Create two random matrices in tensor of shape (16, 4, 9)
# pred = np.arange(36, dtype=np.float32).reshape(4, 9)
# target = np.arange(36, 72, dtype=np.float32).reshape(4, 9)
# pred = np.tile(pred, (16, 1, 1))
# target = np.tile(target, (16, 1, 1))

# # Convert the numpy arrays to torch tensors
# pred = torch.tensor(pred)
# target = torch.tensor(target)

# # Print the torch tensors
# print("Torch Tensor 1:\n", pred)
# print("Torch Tensor 2:\n", target)

# loss = F.mse_loss(pred, target, reduction='none')
# ic(loss)
# loss = reduce(loss, 'b ... -> b (...)', 'mean')
# ic(loss)
# loss = loss.mean()
# ic(loss)


# Create an array of shape (20, 3)
data = np.random.rand(20, 3)

# Store the array in a zarr file
zarr_file = 'data.zarr'
z = zarr.open(zarr_file, mode='w', shape=data.shape, dtype=data.dtype)
z[:] = data

# Print the original zarr array
print("Original zarr array:\n", z[:])

# Resize the zarr array to contain only the last 1 of the 20 items
z.resize(1, 3)
z[:] = z[-1:]

# Print the resized zarr array
print("Resized zarr array:\n", z[:])