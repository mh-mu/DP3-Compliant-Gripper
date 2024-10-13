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

success = pyspacemouse.open(dof_callback=pyspacemouse.print_state, button_callback=pyspacemouse.print_buttons)
if success:
    while 1:
        spacemouse_state = pyspacemouse.read()
        trans = np.array([spacemouse_state.y, -spacemouse_state.x, spacemouse_state.z])/[15, 15, 15]
        rot_rad = np.array([spacemouse_state.roll, spacemouse_state.pitch, -spacemouse_state.yaw]) * 5
        rot_vec = Rotation.from_euler('xyz', rot_rad, degrees=True).as_rotvec()

        action = np.concatenate((rot_vec, trans))

        rot_vec = action[:3]
        rot = so3.from_rotation_vector(rot_vec)
        trans = action[3:6].tolist()
        ic(trans)
        ur5.set_EE_transform_delta((rot, trans))

# ur5.set_EE_transform(([-0.16025840643964528,
#                         0.24645393127427948,
#                         -0.9558125877623072,
#                         0.7834939400974105,
#                         0.6207367988912578,
#                         0.028689237230522555,
#                         0.6003786213682663,
#                         -0.7442756789400599,
#                         -0.2925734518753412],
#                        [-0.4027749179710378, -0.1103704310398187, 0.1532136850835169]))