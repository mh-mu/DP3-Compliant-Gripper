import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from natsort import natsorted
from termcolor import cprint
from gymnasium import spaces

import manipulator_learning.sim.envs as manlearn_envs

class ManipulatorLearnEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name,
                 num_points=1024,
                 ):
        super(ManipulatorLearnEnv, self).__init__()

        self.env = getattr(manlearn_envs, task_name)()
    
        self.observation_space = spaces.Dict({
            'combined_img': spaces.Box(
                low=0,
                high=1,
                shape=(6, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 6), # 3 or 6?
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(39, ), #20 originally, but wrong?
                dtype=np.float32
            ),
        })

    def get_visual_obs(self):

        obs_dict = {
            'combined_img': obs_combined_img,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }
        return obs_dict
            
            
    def step(self, action: np.array):

        obs_dict = {
            'combined_img': obs_combined_img,
            'depth': depth,
            'agent_pos': robot_state.astype(np.float32),
            'point_cloud': point_cloud.astype(np.float32),
            'full_state': raw_state.astype(np.float32),
        }
        
        return obs_dict, reward, done, env_info

    def reset(self):
        obs_dict = {
            'combined_img': obs_combined_img,
            'depth': depth,
            'agent_pos': robot_state.astype(np.float32),
            'point_cloud': point_cloud.astype(np.float32),
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

