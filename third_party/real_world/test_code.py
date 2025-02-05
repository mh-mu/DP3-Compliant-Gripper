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

# Create two rotation matrices using scipy.spatial.transform.Rotation
rotation1 = Rotation.from_euler('xyz', [13, 52, 53], degrees=True).as_matrix()
rotation2 = Rotation.from_euler('xyz', [7, 43, 6], degrees=True).as_matrix()

# Flatten the rotation matrices to lists, column major form
r1 = rotation1.T.flatten().tolist()
r2 = rotation2.T.flatten().tolist()

# Concatenate r1 and r2 into an array of shape (n, 9)
rotations = np.array([r1, r2])
ic(rotations)

rotation_matrices = rotations.reshape(-1, 3, 3, order='F')
ic(rotation_matrices)
rotation_matrices = rotation_matrices.astype(np.float32)
ic(rotation_matrices)
rotation_diff_matrices = np.array([r2 @ np.linalg.inv(r1) for r1, r2 in tqdm(zip(rotation_matrices[:-1], rotation_matrices[1:]), total=len(rotation_matrices) - 1)], dtype=np.float32)
ic(rotation_diff_matrices)

reverse = np.linalg.inv(rotation_diff_matrices) @ rotation2
ic(reverse)
ic(rotation1)

# Convert the rotation difference matrices to Euler angles
rotation_diff_euler = np.array([Rotation.from_matrix(r).as_euler('xyz', degrees=True) for r in rotation_diff_matrices], dtype=np.float32)
ic(rotation_diff_euler)

ic(rotation_diff_matrices)
rotation_diff_matrices = rotation_diff_matrices[:, :-1, :]  # Remove the last row of each 3 by 3 rotation matrix
ic(rotation_diff_matrices)
rotation_diff = rotation_diff_matrices.reshape(-1, 6)
ic(rotation_diff)