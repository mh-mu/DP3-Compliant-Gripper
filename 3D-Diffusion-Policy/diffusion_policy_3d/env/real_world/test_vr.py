import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import time
import cv2

from natsort import natsorted
from termcolor import cprint
from gymnasium import spaces
import pyspacemouse
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..', 'third_party', 'UR5_IMPEDANCE')))

from ur5_controller_wrapper import ur5ControlWrapper
from scipy.spatial.transform import Rotation
from klampt.math import so3, se3
from icecream import ic 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..', 'third_party', 'UR5_Teleop')))

from ur5_controller_wrapper import ur5ControlWrapper
from vive_controller_teleop import *

t_pickup_high = [-0.485, -0.19, 0.22] 
t_pickup = [-0.485, -0.19, 0.13] 
# t_wipe_start = [0.4, -0.02, 0.28]
t_random_start = [-0.585, 0.00, 0.155] # this is for 35 mm #[-0.585, 0.00, 0.1615]
t_fixed_start = [-0.56, 0.0075, 0.13]
R_default = [0, 0, -1, math.sqrt(2)/2, math.sqrt(2)/2, 0, math.sqrt(2)/2, -math.sqrt(2)/2, 0]
ft_ip = 'http://192.168.0.102:80'
gripper_port = '/dev/ttyUSB0'
finger_zero_positions = [0.0675, 0.17] #[0.0725, 0.175] #[0.075, 0.18] #[0.065, 0.175] #[0.0775, 0.1845] #[0.08, 0.1875] #[0.085, 0.19] (2nd, current calibration pic) #[0.095, 0.2] (initial)

UR5_ip = '192.168.0.101'
UR5_home_position = (R_default, t_pickup)
R_EE_WORLD_HOME = [0,0,-1, math.sqrt(2)/2,math.sqrt(2)/2,0,math.sqrt(2)/2,-math.sqrt(2)/2,0] #klampt format
R_ATI_EE = [0, math.sqrt(2)/2,math.sqrt(2)/2, 0,math.sqrt(2)/2,-math.sqrt(2)/2, 1, 0, 0]
HOME_t_obj = [-0.6, 0, 0.08] #[-0.5, 0, 0.08]

ur5 = ur5ControlWrapper(home_T=(R_EE_WORLD_HOME, HOME_t_obj) , ip=UR5_ip, ft_sensor=None)
ur5.set_EE_transform(UR5_home_position)
time.sleep(1)

init_openvr()
init_controllers()
time.sleep(1)
print("Vive Ready")

print("UR5 Initialized")

# prev_pose = get_controller_pose()
# for xz in range(10000):
#     pose = get_controller_pose()
    
#     delta_pose = get_controller_pose_delta(pose, prev_pose)
    
#     prev_pose = pose
#     if is_trigger_active():
#         ur5.move_to_pose_original(delta_pose)
#     time.sleep(0.1)

# openvr.shutdown()
# ur5.close()

prev_pose = get_controller_pose()
for xz in range(10000):
    pose = get_controller_pose()
    
    delta_pose = get_controller_pose_delta(pose, prev_pose)

    rot_vec = Rotation.from_matrix(delta_pose[:3, :3]).as_rotvec()
    delta_trans = delta_pose[:3, 3]
    delta_rot = so3.from_rotation_vector(rot_vec)

    delta_pose = (delta_rot, delta_trans)
    
    prev_pose = pose
    if is_trigger_active():
        ur5.set_EE_transform_delta(delta_pose)
    time.sleep(0.1)

openvr.shutdown()
ur5.close()