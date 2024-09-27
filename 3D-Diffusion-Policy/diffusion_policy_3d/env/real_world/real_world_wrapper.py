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
from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling

current_dir = os.path.dirname(os.path.abspath(__file__))
comp_sim_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, comp_sim_dir)
from src.utils.utils import CompliantGripper

TASK_BOUDNS = {
    'default': [-0.5, -1.5, -0.795, 1, -0.4, 100],
}

class RealWorldEnv(gym.Env):

    def __init__(self, task_name, device="cuda:0", 
                 ):
        super(RealWorldEnv, self).__init__()
    
        self.episode_length = self._max_episode_steps = 100
        self.act_dim = 6
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_dim,),
            dtype=np.float64
        )
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        self.image_size = 128
        self.observation_space = spaces.Dict({
            'wrist_img': spaces.Box(
                low=0,
                high=1,
                shape=(6, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'forces': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(6, ),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(39, ),  # TODO: what dimension
                dtype=np.float32
            ),
        })

    def get_robot_state(self):
        # TODO: get end effector position, get finger motor positions
        eef_pos = None
        finger_right, finger_left = None
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        # TODO: get img from wrist cam
        img = None
        return img

    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_pixels = obs_pixels.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': obs_pixels,
            'agent_pos': robot_state,
        }
        return obs_dict
            
            
    def step(self, action: np.array):

        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()

        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_pixels = obs_pixels.astype(np.float32) / 255 # normalize

        obs_dict = {
            'wrist_img': obs_pixels,
            'agent_pos': robot_state.astype(np.float32),
            'full_state': raw_state.astype(np.float32),
        }

        done = done or self.cur_step >= self.episode_length
        
        return obs_dict, reward, done, env_info

    def reset(self):
        # # added for gymnasium
        # super().reset(seed=seed)

        self.env.reset()
        self.env.reset_model()
        raw_obs = self.env.reset()
        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_pixels = obs_pixels.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': obs_pixels,
            'agent_pos': robot_state.astype(np.float32),
            'full_state': raw_obs.astype(np.float32),
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

