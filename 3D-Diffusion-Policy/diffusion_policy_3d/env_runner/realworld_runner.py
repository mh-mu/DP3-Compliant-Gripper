import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.env import RealWorldEnv
from diffusion_policy_3d.env.real_world.real_world_replay_wrapper import RealWorldReplayEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

import cv2
import os
import pickle
from icecream import ic

class RealworldRunner(BaseRunner):
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
                 use_point_crop=True,
                 num_points=512,
                 interpolate_steps=0,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    RealWorldEnv(task_name=task_name, finger_type='rigid', device=device, demo_device='vr', mode='eval')),
                    # RealWorldEnv(task_name=task_name, finger_type='compliant', device=device, demo_device='vr', mode='eval')),
                    # RealWorldReplayEnv(task_name=task_name, device=device, mode='eval')), # for testing training data
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

        # Determine the filename with increment
        base_path = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/third_party/real_world/rollout_data'
        file_index = 0
        while os.path.exists(os.path.join(base_path, f'obs_dict_list_{file_index}.pkl')):
            file_index += 1
        self.obs_file_path = os.path.join(base_path, f'obs_dict_list_{file_index}.pkl')

        actions_file_index = 0
        while os.path.exists(os.path.join(base_path, f'actions_list_{actions_file_index}.txt')):
            actions_file_index += 1
        self.actions_file_path = os.path.join(base_path, f'actions_list_{actions_file_index}.txt')

    def run(self, policy: BasePolicy, save_video=True, use_force=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        actions_list = []
        obs_dict_list = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in realworld {self.task_name} Realworld Env", leave=False, mininterval=self.tqdm_interval_sec):
            
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
                    # ic()
                    # ic(action_dict['action'].shape)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                # ic()
                # ic(action.shape)

                obs_dict_list.append({k: v.cpu().numpy() for k, v in obs_dict_input.items()})
                
                with open(self.obs_file_path, 'wb') as f:
                    pickle.dump(obs_dict_list, f)

                if self.fps == 30:
                    obs, reward, done, info = env.step(action)
                elif self.fps == 10:
                    # ic(action.shape)
                    obs, reward, done, info = env.step_interpolate(action, interpolate_steps=2)
                    # ic(obs['state'].shape)
                else:
                    raise ValueError(f"fps {self.fps} not supported")

                actions_list.append(action.tolist())
                
                with open(self.actions_file_path, 'w') as f:
                    for action in actions_list:
                        f.write("%s\n" % action)

                # traj_reward += reward
                done = np.all(done)
                is_success = is_success #or max(info['success'])

            all_success_rates.append(is_success)
            # all_traj_rewards.append(traj_reward)

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        # log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        _ = env.reset()

        return log_data
