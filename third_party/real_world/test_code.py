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


# Example usage of the klampt.model.trajectory library

# Create a trajectory with a list of milestones (waypoints)
milestones = [
    [0, 0, 0],
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3]
]

# Create a Trajectory object
traj = trajectory.Trajectory(milestones=milestones)

# Print the trajectory
print("Trajectory milestones:", traj.milestones)

# Evaluate the trajectory at a specific time
time = 1.5
point = traj.eval(time)
print(f"Point at time {time}:", point)

# # Save the trajectory to a file
# traj.save("trajectory.path")

# # Load the trajectory from a file
# loaded_traj = trajectory.Trajectory()
# loaded_traj.load("trajectory.path")
# print("Loaded trajectory milestones:", loaded_traj.milestones)