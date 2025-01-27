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
from .T42_controller import T42_controller
from . import CONSTANTS
from scipy.spatial.transform import Rotation
from klampt.math import so3, se3
from icecream import ic 
import pickle

class RealWorldEnv(gym.Env):

    def __init__(self, task_name, demo_device, finger_type, device="cuda:0", mode='train',
                 ):
        super(RealWorldEnv, self).__init__()
    
        self.episode_length = self._max_episode_steps = 600
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

        self.demo_device = demo_device
        if self.demo_device not in ('spacemouse', 'vr'):
            raise ValueError(f'Unrecognized demo device: {demo_device}. Device should be "spacemouse" or "vr".')

        self.finger_type = finger_type
        if self.finger_type == 'compliant':
            self.gripper = T42_controller(CONSTANTS.finger_offset_positions_compliant, finger_type='compliant', port=CONSTANTS.gripper_port, data_collection_mode=False)
        elif self.finger_type == 'rigid':
            self.gripper = T42_controller(CONSTANTS.finger_offset_positions_rigid, finger_type='rigid', port=CONSTANTS.gripper_port, data_collection_mode=False)
        else:
            raise ValueError(f'Unrecognized finger type. Finger type should be compliant or rigid, got {finger_type} instead.')
        
        if self.mode == 'train':
            self.step_frequency = 30
        elif self.mode == 'eval':
            self.step_frequency = 10 # TODO: change this
        self.step_period = 1 / self.step_frequency
        # self.target_trans_speed = 10
        # self.target_rot_speed = 15
        self.target_trans_speed = 2e1
        self.target_rot_speed = 3e1
        # self.target_trans_speed = 2e2
        # self.target_rot_speed = 3e2
        if self.demo_device == 'spacemouse':
            self.trans_scale = 14 * self.step_period
            self.rot_scale = 1e3 * self.step_period
        elif self.demo_device == 'vr':
            self.trans_scale = self.target_trans_speed / self.step_frequency
            self.rot_scale = self.target_rot_speed / self.step_frequency

        self.cap_wrist = cv2.VideoCapture(2)
        self.cap_gripper = cv2.VideoCapture(5)
        self.cap_third_view = cv2.VideoCapture(9)
        ic('started cameras')
        if not self.cap_wrist.isOpened():
            print("Error: Could not open wrist camera.")
            exit()
        if not self.cap_gripper.isOpened():
            print("Error: Could not open gripper camera.")
            exit()
        if not self.cap_third_view.isOpened():
            print("Error: Could not open third view camera.")
            exit()
        
        self.ur5_action_list = []
        self.force_list = []

        if self.mode == 'train':
            # Create a window to display forces in real-time
            self.force_window_name = "Real-Time Forces"
            cv2.namedWindow(self.force_window_name, cv2.WINDOW_NORMAL)
            self.force_display_img = np.zeros((300, 600, 3), dtype=np.uint8)

    def get_robot_state(self):
        '''
        13 elements, orientation(9) and ee position(3), fingers open or closed (1)
        '''
        eef_pos = self.ur5_controller.get_EE_transform()
        # finger_positions, _ = self.gripper.read_motor_positions() # TODO: change to scaled gripper value
        return np.concatenate([np.array(eef_pos[0] + eef_pos[1]), np.array([self.gripper_state])])

    def get_rgb(self):
        ret_wrist, img_wrist = self.cap_wrist.read()
        ret_gripper, img_gripper = self.cap_gripper.read()
        ret_third_view, img_third_view = self.cap_third_view.read()
        if not ret_wrist:
            print("Error: Could not read wrist camera frame.")
        if not ret_gripper:
            print("Error: Could not read gripper camera frame.")
        if not ret_third_view:
            print("Error: Could not read third view camera frame.")
        return img_wrist, img_gripper, img_third_view
    
    def get_robot_force(self):
        force = self.ur5_controller.get_EE_wrench()[0:3]
        return np.array(force)

    def get_visual_obs(self):
        img_wrist, img_gripper, img_third_view = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if img_wrist.shape[0] != 3:
            img_wrist = img_wrist.transpose(2, 0, 1)
        if img_gripper.shape[0] != 3:
            img_gripper = img_gripper.transpose(2, 0, 1)
        if img_third_view.shape[0] != 3:
            img_third_view = img_third_view.transpose(2, 0, 1)

        img_wrist = img_wrist.astype(np.float32) / 255
        img_gripper = img_gripper.astype(np.float32) / 255
        img_third_view = img_third_view.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': img_wrist,
            'gripper_img': img_gripper,
            'third_view_img': img_third_view,
            'force': robot_force,
            'state': robot_state,
        }
        return obs_dict
            
            
    def step(self, action: np.array):
        start_time = time.time()

        # perform actions
        rot_vec = action[:3]
        trans = action[3:6].tolist()

        # cap rotation and translation actions
        if np.any(np.abs(rot_vec) > 0.02) or np.any(np.abs(trans) > 0.005):
            rot_vec = np.zeros_like(rot_vec)
            trans = np.zeros_like(trans)
        
        rot = so3.from_rotation_vector(rot_vec)

        # # record the previous action performed by the UR5 (for debugging)
        # current_pose = self.ur5_controller.get_EE_transform()
        # delta_position = list(np.array(current_pose[1]) - np.array(self.previous_pose[1]))
        # self.ur5_action_list.append(delta_position)
        # with open('ur5_action_list.pkl', 'wb') as f:
        #     pickle.dump(self.ur5_action_list, f)
        # self.previous_pose = current_pose


        if self.mode == 'train':
            self.ur5_controller.set_EE_transform_delta((rot, trans))
        elif self.mode == 'eval':
            # print('The predicted action is:', rot_vec, trans)
            # print('Execute action? (y/n)')
            # if input() == 'y':
            #     self.ur5_controller.set_EE_transform_delta((rot, trans))
            # else:
            #     print('Action not executed. Exiting...')
            #     exit()
            self.ur5_controller.set_EE_transform_delta((rot, trans), max_trans_v=0.2, max_rot_v=0.1)
        
        # gripper_action = action[-1]
        # if gripper_action != self.prev_gripper_pos: 
        #     self.prev_gripper_pos = gripper_action
        #     if gripper_action == CONSTANTS.CLOSE:
        #         self.gripper.close()
        #         self.gripper_state = CONSTANTS.CLOSE
        #     elif gripper_action == CONSTANTS.OPEN:
        #         self.gripper.release()
        #         self.gripper_state = CONSTANTS.OPEN

        self.cur_step += 1

        img_wrist, img_gripper, img_third_view = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if self.mode == 'train':
            # display realtime force values for demo collection
            self.force_display_img.fill(0)
            bar_width = 50
            max_force = np.max(np.abs(robot_force)) + 1
            for i, force in enumerate(robot_force):
                color = (255, 255, 0) if i == 0 else (0, 255, 0) if i == 1 else (0, 255, 255)
                start_point = (i * 200 + 100, 150)
                end_point = (i * 200 + 100, 150 - int(force / max_force * 100))
                cv2.line(self.force_display_img, start_point, end_point, color, bar_width)
                cv2.putText(self.force_display_img, f'{force:.2f}', (i * 200 + 75, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.imshow(self.force_window_name, self.force_display_img)
            cv2.waitKey(1)

        # ic(robot_force)
        # record the forces during rollout (for debugging)
        self.force_list.append(robot_force)
        with open('force_list.pkl', 'wb') as f:
            pickle.dump(self.force_list, f)

        if img_wrist.shape[0] != 3:
            img_wrist = img_wrist.transpose(2, 0, 1)
        if img_gripper.shape[0] != 3:
            img_gripper = img_gripper.transpose(2, 0, 1)
        if img_third_view.shape[0] != 3:
            img_third_view = img_third_view.transpose(2, 0, 1)

        img_wrist = img_wrist.astype(np.float32) / 255
        img_gripper = img_gripper.astype(np.float32) / 255
        img_third_view = img_third_view.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': img_wrist,
            'gripper_img': img_gripper,
            'third_view_img': img_third_view,
            'force': robot_force,
            'state': robot_state,
        }

        done = self.cur_step >= self.episode_length

        elapsed_time = time.time() - start_time
        sleep_time = self.step_period - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        return obs_dict, None, done, None

    def reset(self, seed = None, options = None):
        self.ur5_controller = ur5ControlWrapper(home_T = (CONSTANTS.R_EE_WORLD_HOME, CONSTANTS.HOME_t_obj) , ip = CONSTANTS.UR5_ip, ft_sensor=None)
        time.sleep(2)

        self.ur5_controller.set_EE_transform_linear(CONSTANTS.UR5_home_position, max_trans_v = 0.8)
        self.gripper.close()
        self.gripper_state = CONSTANTS.CLOSE
        self.prev_gripper_pos = CONSTANTS.CLOSE
        self.ur5_controller.zero_ft_sensor()

        # self.previous_pose = self.ur5_controller.get_EE_transform()
        # with open('ur5_action_list.pkl', 'wb') as f:
        #     pickle.dump(self.ur5_action_list, f)

        with open('force_list.pkl', 'wb') as f:
            pickle.dump(self.force_list, f)

        self.cur_step = 0

        img_wrist, img_gripper, img_third_view = self.get_rgb()
        robot_state = self.get_robot_state()
        robot_force = self.get_robot_force()

        if img_wrist.shape[0] != 3:
            img_wrist = img_wrist.transpose(2, 0, 1)
        if img_gripper.shape[0] != 3:
            img_gripper = img_gripper.transpose(2, 0, 1)
        if img_third_view.shape[0] != 3:
            img_third_view = img_third_view.transpose(2, 0, 1)

        img_wrist = img_wrist.astype(np.float32) / 255
        img_gripper = img_gripper.astype(np.float32) / 255
        img_third_view = img_third_view.astype(np.float32) / 255

        obs_dict = {
            'wrist_img': img_wrist,
            'gripper_img': img_gripper,
            'third_view_img': img_third_view,
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

