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

from klampt.math import so3, se3
from icecream import ic 
import pickle

from .dummy_env import DummyEnv

class DummyRealWorldEnv(gym.Env):

    def __init__(self, device="cuda:0", mode='train',
                 ):
        super(DummyRealWorldEnv, self).__init__()
    
        self.episode_length = self._max_episode_steps = 30
        self.mode = mode
        self.act_dim = 2
        self.action_space = spaces.Box(
            low=-320,
            high=320,
            shape=(self.act_dim,),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            'wrist_img': spaces.Box(
                low=0,
                high=1,
                shape=(3, 480, 640),
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

        self.env = DummyEnv(goal_position=np.array([20, 60]), force_coefficient=1e-2)
        
        if self.mode == 'eval':
            self.ur5_action_list = []
            self.force_list = []
            save_dir = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/third_party/real_world/rollout_data'
            os.makedirs(save_dir, exist_ok=True)
            file_index = 0
            self.save_path = os.path.join(save_dir, f'force_list_{file_index}.pkl')
            while os.path.exists(self.save_path):
                file_index += 1
                self.save_path = os.path.join(save_dir, f'force_list_{file_index}.pkl')


    def get_robot_state(self):
        '''
        2 elements, relative movement from last position
        '''
        return self.env.get_latest_action()

    def get_rgb(self):
        return self.env.render()
    
    def get_robot_force(self):
        return self.env.get_force()

    def get_visual_obs(self):
        img_wrist = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if img_wrist.shape[0] != 3:
            img_wrist = img_wrist.transpose(2, 0, 1)

        img_wrist = img_wrist.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': img_wrist,
            'force': robot_force,
            'state': robot_state,
        }
        return obs_dict
            
            
    def step(self, action: np.array):
        
        done = self.env.step(action)
        self.cur_step += 1

        img_wrist = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if img_wrist.shape[0] != 3:
            img_wrist = img_wrist.transpose(2, 0, 1)

        img_wrist = img_wrist.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': img_wrist,
            'force': robot_force,
            'state': robot_state,
        }

        done = self.cur_step >= self.episode_length
        
        return obs_dict, 0, done, 0

    def reset(self, seed = None, options = None):

        self.env.reset()
        self.cur_step = 0

        img_wrist = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if img_wrist.shape[0] != 3:
            img_wrist = img_wrist.transpose(2, 0, 1)

        img_wrist = img_wrist.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': img_wrist,
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

