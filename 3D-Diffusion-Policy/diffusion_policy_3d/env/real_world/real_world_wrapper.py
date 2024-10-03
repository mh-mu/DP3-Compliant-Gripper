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

from third_party.UR5_IMPEDANCE.ur5_controller_wrapper import ur5ControlWrapper
from .T42_controller import T42_controller
from . import CONSTANTS
from scipy.spatial.transform import Rotation
from klampt.math import so3, se3

class RealWorldEnv(gym.Env):

    def __init__(self, task_name, device="cuda:0", 
                 ):
        super(RealWorldEnv, self).__init__()
    
        self.episode_length = self._max_episode_steps = 200
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

        self.ur5_controller = ur5ControlWrapper(home_T = (CONSTANTS.R_EE_WORLD_HOME, CONSTANTS.HOME_t_obj) , ip = CONSTANTS.ur5_ip,
                        ft_sensor=None)
        self.gripper = T42_controller(CONSTANTS.finger_zero_positions, port=CONSTANTS.gripper_port, data_collection_mode=False)
        

    def get_robot_state(self):
        '''
        8 elements, ee position and orientation(6), finger motor positions(2)
        '''
        eef_pos = self.ur5_controller.get_ee_position()
        finger_positions, _ = self.gripper.read_motor_positions()
        return np.concatenate([eef_pos, finger_positions])

    def get_rgb(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()
        ret, img = cap.read()
        if not ret:
            print("Error: Could not read frame.")
        cap.release()
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

        # perform actions
        rot_rad = action[:3]
        rot = Rotation.from_euler('xyz', rot_rad, degrees=True).as_matrix()
        rot = so3.from_matrix(rot)
        trans = action[2:6]
        trans = se3.translation(trans)
        self.ur5_controller.set_EE_transform_delta((rot, trans))
        gripper_action = action[-1]
        if gripper_action == CONSTANTS.CLOSE:
            self.gripper.close()
        elif gripper_action == CONSTANTS.OPEN:
            self.gripper.release()

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
        
        return obs_dict, None, done, None

    def reset(self):
        # # added for gymnasium
        # super().reset(seed=seed)

        self.ur5_controller.set_EE_transform() # TODO: go to home position
        self.gripper.release()

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

