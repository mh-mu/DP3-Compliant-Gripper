import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import metaworld
import random
import time
import cv2

from natsort import natsorted
from termcolor import cprint
from gym import spaces
from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling

current_dir = os.path.dirname(os.path.abspath(__file__))
comp_sim_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, comp_sim_dir)
from src.utils.utils import CompliantGripper

TASK_BOUDNS = {
    'default': [-0.5, -1.5, -0.795, 1, -0.4, 100],
}

class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0", 
                 use_point_crop=True,
                 num_points=1024,
                 ):
        super(MetaWorldEnv, self).__init__()

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False

        # https://arxiv.org/abs/2212.05698
        # self.env.sim.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]

        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5
        
        self.device_id = int(device.split(":")[-1])
        
        self.image_size = 128
        
        self.pc_generator = PointCloudGenerator(sim=self.env.sim, cam_names=['corner2'], img_size=self.image_size)
        self.use_point_crop = use_point_crop
        cprint("[MetaWorldEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        self.num_points = num_points # 512
        
        x_angle = 61.4
        y_angle = -7
        self.pc_transform = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(x_angle)), np.sin(np.deg2rad(x_angle))],
            [0, -np.sin(np.deg2rad(x_angle)), np.cos(np.deg2rad(x_angle))]
        ]) @ np.array([
            [np.cos(np.deg2rad(y_angle)), 0, np.sin(np.deg2rad(y_angle))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(y_angle)), 0, np.cos(np.deg2rad(y_angle))]
        ])
        
        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        if task_name in TASK_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[task_name]
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]
        
    
        self.episode_length = self._max_episode_steps = 200
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        # Compliant gripper 
        self.compliant_gripper_urdf_path = os.path.join(comp_sim_dir, 'src/utils/FFF.urdf')
        gripper_k_dict = {
            # 'hammer-v2-goal-observable': [7e3, 1.4e4],
            # 'assembly-v2-goal-observable': [1.5e3, 1.2e4],
            # 'push-v2-goal-observable': [6e3, 4e3],
            # 'dial-turn-v2-goal-observable': [1.2e4, 1.5e4],
            # 'button-press-topdown-v2-goal-observable': [6e4, 1.4e4],
            # 'drawer-close-v2-goal-observable': [5e4, 7e3],
            # 'handle-pull-side-v2-goal-observable': [1.2e4, 2.5e3],
            # 'pick-place-v2-goal-observable': [5e3, 7e3],
            # 'drawer-close-v2-goal-observable': [5e3, 7e3],
            # 'push-back-v2-goal-observable': [5e3, 7e3],
            # 'push-v2-goal-observable': [5e3, 7e3],
            # 'pick-out-of-hole-v2-goal-observable': [400, 400],
            # 'box-close-v2-goal-observable': [4e4, 1.4e4],
            # 'drawer-open-v2-goal-observable': [4e4, 1.4e4],
            'drawer-open-v2-goal-observable': [1.8e4, 8e3],
        }
        if task_name not in gripper_k_dict:
            raise KeyError(f"Compliant gripper K is not defined for task '{task_name}'")
        self.compliant_gripper = CompliantGripper(self.compliant_gripper_urdf_path, K=gripper_k_dict[task_name])
        self.gripper_forces = []
    
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
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20, ),
                dtype=np.float32
            ),
        })

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV', 'implicitForceView')
        # img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name="corner3", device_id=self.device_id)
        img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name="implicitForceView", device_id=self.device_id)
        return img
    
    def get_ee_contact_forces(self):
        '''
        Calculate contact force of gripper from mujoco based on contact locations (Mei)
        Arg(s):
            None
        Returns:
            left_forces : np[(3, )]
                Forces on the left finger in the contact frame
            right_forces : np[(3, )]
                Forces on the right finger in the contact frame
        '''
        left_forces = self.env.get_body_contact_force('leftpad') + self.env.get_body_contact_force('leftclaw')
        right_forces = self.env.get_body_contact_force('rightpad') + self.env.get_body_contact_force('rightclaw')
        return left_forces, right_forces
    
    def render_compliant_image(self, left_finger_forces, right_finger_forces):
        '''
        Create compliant gripper object and render image.
        Arg(s):
            left_finger_forces : np.array(3, )
                Forces in the x, y and z direction for the left gripper finger
            right_finger_forces : np.array(3, )
                Forces in the x, y and z direction for the right gripper finger
        Returns:
            image : PIL.Image
                rendered image of the gripper under forces
        '''
        # calculate combined external force
        extern_force = left_finger_forces[[0, 1]] + right_finger_forces[[0, 1]]
        #cprint(f'Gripper force: {left_finger_forces + right_finger_forces}', 'yellow')
        self.gripper_forces.append(left_finger_forces + right_finger_forces)

        return self.compliant_gripper.render_img(extern_force)

    def render_high_res(self, resolution=1024):
        img = self.env.sim.render(width=resolution, height=resolution, camera_name="corner2", device_id=self.device_id)
        return img
    

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(device_id=self.device_id) # raw point cloud, Nx3
        
        
        if not use_rgb:
            point_cloud = point_cloud[..., :3]
        
        
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset
        
        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
        
        depth = depth[::-1]
        
        return point_cloud, depth
        

    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        # add compliant gripper image
        l_forces, r_forces = self.get_ee_contact_forces()
        compliant_img = self.render_compliant_image(l_forces, r_forces)
        compliant_img = np.asarray(compliant_img) # h * w * c
        # make compliant image square
        obs_compliant_img = np.ones((640, 640, 3))*255
        obs_compliant_img[140:500, :, :] = compliant_img
        
        # combine rgb image with compliant image
        _, h, w = obs_pixels.shape # c * h * w
        resized_obs_compliant_img = cv2.resize(obs_compliant_img, (h, w), interpolation=cv2.INTER_AREA)
        resized_obs_compliant_img = np.transpose(resized_obs_compliant_img, (2,0,1)) # c * h * w
        obs_combined_img = np.concatenate((obs_pixels, resized_obs_compliant_img), axis=0)
        obs_combined_img = obs_combined_img.astype(np.float32) / 255

        obs_dict = {
            'combined_img': obs_combined_img,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }
        return obs_dict
            
            
    def step(self, action: np.array):

        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1

        #cprint(f'\n\n\n\nCurrent Step: {self.cur_step}', 'green')

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        # add compliant gripper image
        l_forces, r_forces = self.get_ee_contact_forces()
        compliant_img = self.render_compliant_image(l_forces, r_forces)
        compliant_img = np.asarray(compliant_img) # h * w * c
        # make compliant image square
        obs_compliant_img = np.ones((640, 640, 3))*255
        obs_compliant_img[140:500, :, :] = compliant_img
        
        # combine rgb image with compliant image
        _, h, w = obs_pixels.shape # c * h * w
        resized_obs_compliant_img = cv2.resize(obs_compliant_img, (h, w), interpolation=cv2.INTER_AREA)
        resized_obs_compliant_img = np.transpose(resized_obs_compliant_img, (2,0,1)) # c * h * w
        obs_combined_img = np.concatenate((obs_pixels, resized_obs_compliant_img), axis=0)
        obs_combined_img = obs_combined_img.astype(np.float32) / 255

        obs_dict = {
            'combined_img': obs_combined_img,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_state,
        }

        done = done or self.cur_step >= self.episode_length
        
        return obs_dict, reward, done, env_info

    def reset(self):
        self.env.reset()
        self.env.reset_model()
        raw_obs = self.env.reset()
        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        # add compliant gripper image
        l_forces, r_forces = self.get_ee_contact_forces()
        compliant_img = self.render_compliant_image(l_forces, r_forces)
        compliant_img = np.asarray(compliant_img) # h * w * c
        # make compliant image square
        obs_compliant_img = np.ones((640, 640, 3))*255
        obs_compliant_img[140:500, :, :] = compliant_img
        
        # combine rgb image with compliant image
        _, h, w = obs_pixels.shape # c * h * w
        resized_obs_compliant_img = cv2.resize(obs_compliant_img, (h, w), interpolation=cv2.INTER_AREA)
        resized_obs_compliant_img = np.transpose(resized_obs_compliant_img, (2,0,1)) # c * h * w
        obs_combined_img = np.concatenate((obs_pixels, resized_obs_compliant_img), axis=0)
        obs_combined_img = obs_combined_img.astype(np.float32) / 255
        
        obs_dict = {
            'combined_img': obs_combined_img,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_obs,
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

