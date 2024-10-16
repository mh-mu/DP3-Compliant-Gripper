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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..', 'third_party', 'UR5_IMPEDANCE')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..', 'third_party', 'UR5_Teleop')))

from ur5_controller_wrapper import ur5ControlWrapper
from vive_controller_teleop import *
# from .T42_controller import T42_controller
from . import CONSTANTS
from scipy.spatial.transform import Rotation
from klampt.math import so3, se3
from icecream import ic 

class RealWorldEnv(gym.Env):

    def __init__(self, task_name, demo_device, device="cuda:0", 
                 ):
        super(RealWorldEnv, self).__init__()
    
        self.episode_length = self._max_episode_steps = 10000
        self.act_dim = 7
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_dim,),
            dtype=np.float64
        )
        self.image_size = 128
        self.observation_space = spaces.Dict({
            'wrist_img': spaces.Box(
                low=0,
                high=1,
                shape=(6, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'force': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3, ),
                dtype=np.float32
            ),
            'state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(11, ),
                dtype=np.float32
            ),
        })

        self.demo_device = demo_device
        if self.demo_device not in ('spacemouse', 'vr'):
            raise ValueError(f'Unrecognized demo device: {demo_device}. Device should be "spacemouse" or "vr".')

        # self.cap = cv2.VideoCapture(2) # debug
        # if not self.cap.isOpened():
        #     print("Error: Could not open webcam.")
        #     exit()
        self.ur5_controller = ur5ControlWrapper(home_T = (CONSTANTS.R_EE_WORLD_HOME, CONSTANTS.HOME_t_obj) , ip = CONSTANTS.UR5_ip, ft_sensor=None)
        # self.gripper = T42_controller(CONSTANTS.finger_zero_positions, port=CONSTANTS.gripper_port, data_collection_mode=False) # debug
        self.step_frequency = 500
        self.step_period = 1 / self.step_frequency

        if self.demo_device == 'spacemouse':
            self.trans_scale = 14 * self.step_period
            self.rot_scale = 1e3 * self.step_period
        elif self.demo_device == 'vr':
            # self.trans_scale = 1e3 * self.step_period # TODO: set scaling value
            # self.rot_scale = 1e3 * self.step_period
            self.trans_scale = 10
            self.rot_scale = 10

    def get_robot_state(self):
        '''
        8 elements, ee position and orientation(6), finger motor positions(2)
        '''
        eef_pos = self.ur5_controller.get_EE_transform()
        # finger_positions, _ = self.gripper.read_motor_positions() # debug
        finger_positions = np.zeros((2,)) # debug
        return np.concatenate([np.array(eef_pos[0] + eef_pos[1]), finger_positions])

    def get_rgb(self):
        # ret, img = self.cap.read()
        img = np.zeros((3, 128, 128)) # debug
        # if not ret:
        #     print("Error: Could not read webcam frame.")
        return img
    
    def get_robot_force(self):
        force = self.ur5_controller.get_EE_wrench()[0:3]
        return np.array(force)

    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_pixels = obs_pixels.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': obs_pixels,
            'force': robot_force,
            'state': robot_state,
        }
        return obs_dict
            
            
    def step(self, action: np.array):
        start_time = time.time()

        # perform actions
        rot_vec = action[:3]
        rot = so3.from_rotation_vector(rot_vec)
        trans = action[3:6].tolist()

        # self.ur5_controller.set_EE_transform_delta((rot, trans))
        self.ur5_controller.move_to_pose((rot, trans))
        
        gripper_action = action[-1]
        # if gripper_action != self.prev_gripper_pos: # debug
        #     self.prev_gripper_pos = gripper_action
        #     if gripper_action == CONSTANTS.CLOSE:
        #         self.gripper.close()
        #     elif gripper_action == CONSTANTS.OPEN:
        #         self.gripper.release()

        self.cur_step += 1

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_pixels = obs_pixels.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': obs_pixels,
            'force': robot_force,
            'state': robot_state,
        }

        done = self.cur_step >= self.episode_length

        elapsed_time = time.time() - start_time
        sleep_time = self.step_period - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        return obs_dict, None, done, None

    def reset(self):
        # self.ur5_controller.set_EE_transform(CONSTANTS.UR5_home_position) # TODO: debugging, return to home position
        # self.gripper.release()
        self.prev_gripper_pos = CONSTANTS.OPEN
        self.ur5_controller.zero_ft_sensor()

        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_pixels = obs_pixels.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': obs_pixels,
            'force': robot_force,
            'state': robot_state,
        }

        return obs_dict

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        img = self.get_rgb()
        return img

    def close(self):
        pass

