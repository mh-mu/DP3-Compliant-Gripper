import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.env import DummyRealWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

import cv2
import os
from icecream import ic 

class DummyRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=30,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 num_points=512
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    DummyRealWorldEnv()),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        self.video_save_dir = f'../../../eval_videos/{self.task_name}/'
        os.makedirs(self.video_save_dir, exist_ok=True)

    def run(self, policy: BasePolicy, save_video=False, use_force=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        
        # for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Dummy {self.task_name} Env", leave=False, mininterval=self.tqdm_interval_sec):

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Dummy {self.task_name} Env", leave=False):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['wrist_img'] = obs_dict['wrist_img'].unsqueeze(0)
                    obs_dict_input['state'] = obs_dict['state'].unsqueeze(0)
                    if use_force:
                        obs_dict_input['force'] = obs_dict['force'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)

                traj_reward += reward
                done = np.all(done)
                # ic()
                # ic(info)
                is_success = is_success or max(info['success'])

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)

            # save video
            videos = np.array(env.env.get_video())
            
            if save_video:
                video_filename = os.path.join(self.video_save_dir, f'{self.task_name}{episode_idx}_{is_success}_.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                num_frames = videos.shape[0]
                _, height, width = videos.shape[1:]

                out = cv2.VideoWriter(video_filename, fourcc, self.fps, (width, height))

                for i in range(videos.shape[0]):
                    frame = videos[i]
                    frame = np.transpose(frame, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
                    out.write(frame)

                # Release video writer
                out.release()

                print(f"Eval video saved as {video_filename}")

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        _ = env.reset()
        videos = None

        return log_data
