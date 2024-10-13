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
from diffusion_policy_3d.env.real_world import CONSTANTS
import keyboard
from icecream import ic 

rot_vec = np.array([0., 0., 0.])
ic(rot_vec)
rot = so3.from_rotation_vector(rot_vec)
ic(rot)