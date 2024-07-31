import gymnasium as gym
import numpy as np
from termcolor import cprint
import cv2


class SimpleVideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='rgb_array',
            steps_per_render=1,
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.steps_per_render = steps_per_render

        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()

        # frame = self.env.render(mode=self.mode)
        frame = self.get_img()
        # assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        self.step_count = 1
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        
        # frame = self.env.render(mode=self.mode)
        frame = self.get_img()
        # assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        return result

    def get_img(self):
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

        return obs_combined_img
    
    def get_video(self):
        video = np.stack(self.frames, axis=0) # (T, H, W, C)
        # to store as mp4 in wandb, we need (T, H, W, C) -> (T, C, H, W)
        # video = video.transpose(0, 3, 1, 2)
        return video

