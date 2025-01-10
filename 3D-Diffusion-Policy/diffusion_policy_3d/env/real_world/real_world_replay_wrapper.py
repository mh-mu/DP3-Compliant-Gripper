import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import time
import cv2
import zarr

from natsort import natsorted
from termcolor import cprint
from gymnasium import spaces

from icecream import ic 

class RealWorldReplayEnv(gym.Env):
    '''
    For replaying training demo actions
    '''

    def __init__(self, task_name, device="cuda:0", mode='train',
                 ):
        super(RealWorldReplayEnv, self).__init__()
    
        self.episode_length = self._max_episode_steps = 300
        self.mode = mode
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

        def read_zarr_folder(folder_path):
            try:
                zarr_data = zarr.open(folder_path, mode='r')
                print(f"Successfully opened Zarr folder: {folder_path}")
                return zarr_data
            except Exception as e:
                print(f"Error opening Zarr folder: {e}")
                return None
            
        folder_path = "/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_line_10Hz_expert.zarr"
        self.zarr_data = read_zarr_folder(folder_path)
        self.cur_step = 0

    def get_robot_state(self):
        '''
        13 elements, orientation(9) and ee position(3), fingers open or closed (1)
        '''
        return self.zarr_data['data/state'][self.cur_step]

    def get_rgb(self):
        return self.zarr_data['data/wrist_img'][self.cur_step]
    
    def get_robot_force(self):
        return self.zarr_data['data/force'][self.cur_step]

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

        self.cur_step += 1
        ic(self.cur_step)

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        # obs_pixels = obs_pixels.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': obs_pixels,
            'force': robot_force,
            'state': robot_state,
        }

        done = self.cur_step >= self.episode_length
        
        return obs_dict, None, done, None

    def reset(self, seed = None, options = None):

        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        # obs_pixels = obs_pixels.astype(np.float32) / 255

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

