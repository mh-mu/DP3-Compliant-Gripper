# bash scripts/metaworld/gen_demonstration_expert.sh reach 5
import argparse
import os
import zarr
import numpy as np
from diffusion_policy_3d.env.real_world import DummyRealWorldEnv
from termcolor import cprint
import copy
import imageio
import cv2
import sys, time

from scipy.spatial.transform import Rotation
from klampt.math import so3, se3
from diffusion_policy_3d.env.real_world import CONSTANTS
import keyboard
from icecream import ic 
import atexit

seed = np.random.randint(0, 100)

def main(args):
	env_name = args.env_name
	
	save_dir = os.path.join(args.root_dir, 'dummy-real-world_'+args.env_name+'_expert.zarr')
	if os.path.exists(save_dir):
		cprint('Data already exists at {}'.format(save_dir), 'red')
		cprint("If you want to overwrite, delete the existing directory first.", "red")
		cprint("Do you want to overwrite? (y/n)", "red")
		user_input = input()
		if user_input == 'y':
			cprint('Overwriting {}'.format(save_dir), 'red')
			os.system('rm -rf {}'.format(save_dir))
		else:
			cprint('Exiting', 'red')
			return
	os.makedirs(save_dir, exist_ok=True)

	e = DummyRealWorldEnv()
	
	num_episodes = args.num_episodes
	cprint(f"Number of episodes : {num_episodes}", "yellow")

	total_count = 0
	episode_ends_arrays = []
	
	episode_idx = 0
	
	# loop over episodes
	while episode_idx < num_episodes:

		e.reset()

		obs_dict = e.get_visual_obs()

		done = False

		wrist_img_arrays_sub = []
		force_arrays_sub = []
		state_arrays_sub = []
		action_arrays_sub = []
		total_count_sub = 0

		zarr_root = zarr.open(save_dir, mode='a')
		zarr_data = zarr_root.require_group('data')
		zarr_meta = zarr_root.require_group('meta')
  
		while not done:
			total_count_sub += 1
			
			obs_wrist_img = obs_dict['wrist_img']
			obs_force = obs_dict['force']
			obs_robot_state = obs_dict['state']

			wrist_img_arrays_sub.append(obs_wrist_img)
			force_arrays_sub.append(obs_force)
			state_arrays_sub.append(obs_robot_state)
			
			action = e.env.calculate_action_towards_goal(initial_action_size=100)
		
			action_arrays_sub.append(action)
			obs_dict, _, done, _ = e.step(action)
   
			if done:
				break

		total_count += total_count_sub

		if episode_idx > 0:
			episode_ends_arrays = episode_ends_arrays.tolist()
		episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode
		cprint('Episode: {}'.format(episode_idx), 'green')
		episode_idx += 1



		###############################
		# save data after each episode
		###############################

		# Convert lists to numpy arrays
		wrist_img_arrays_sub = np.stack(wrist_img_arrays_sub, axis=0)
		if wrist_img_arrays_sub.shape[1] == 3: # make channel last
			wrist_img_arrays_sub = np.transpose(wrist_img_arrays_sub, (0,2,3,1))
		force_arrays_sub = np.stack(force_arrays_sub, axis=0)
		state_arrays_sub = np.stack(state_arrays_sub, axis=0)
		action_arrays_sub = np.stack(action_arrays_sub, axis=0)

		# Append to existing zarr datasets
		if 'wrist_img' not in zarr_data:
			zarr_data.create_dataset('wrist_img', data=wrist_img_arrays_sub, chunks=(100, *wrist_img_arrays_sub.shape[1:]), dtype='float16', compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1))
		else:
			zarr_data['wrist_img'].append(wrist_img_arrays_sub)

		if 'force' not in zarr_data:
			zarr_data.create_dataset('force', data=force_arrays_sub, chunks=(100, force_arrays_sub.shape[1]), dtype='float16', compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1))
		else:
			zarr_data['force'].append(force_arrays_sub)

		if 'state' not in zarr_data:
			zarr_data.create_dataset('state', data=state_arrays_sub, chunks=(100, state_arrays_sub.shape[1]), dtype='float16', compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1))
		else:
			zarr_data['state'].append(state_arrays_sub)

		if 'action' not in zarr_data:
			zarr_data.create_dataset('action', data=action_arrays_sub, chunks=(100, action_arrays_sub.shape[1]), dtype='float16', compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1))
		else:
			zarr_data['action'].append(action_arrays_sub)

		# Save episode ends
		episode_ends_arrays = np.array(episode_ends_arrays)
		if 'episode_ends' not in zarr_meta:
			zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=1))
		else:
			zarr_meta['episode_ends'].append(episode_ends_arrays)

		cprint(f'-'*50, 'cyan')
		# print shape
		cprint(f'wrist img shape: {zarr_data["wrist_img"].shape}, range: [{zarr_data["wrist_img"][:].min()}, {zarr_data["wrist_img"][:].max()}]', 'green')
		cprint(f'force shape: {zarr_data["force"].shape}, range: [{zarr_data["force"][:].min()}, {zarr_data["force"][:].max()}]', 'green')
		cprint(f'state shape: {zarr_data["state"].shape}, range: [{zarr_data["state"][:].min()}, {zarr_data["state"][:].max()}]', 'green')
		cprint(f'action shape: {zarr_data["action"].shape}, range: [{zarr_data["action"][:].min()}, {zarr_data["action"][:].max()}]', 'green')
		cprint(f'Saved zarr file to {save_dir}', 'green')

	# clean up
	del episode_ends_arrays, wrist_img_arrays_sub, force_arrays_sub, state_arrays_sub, action_arrays_sub
	del zarr_root, zarr_data, zarr_meta


 
if __name__ == "__main__":
    
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='test')
	parser.add_argument('--demo_device', type=str, default='vr')
	parser.add_argument('--finger_type', type=str, default='rigid')
	parser.add_argument('--num_episodes', type=int, default=10)
	parser.add_argument('--root_dir', type=str, default="../../3D-Diffusion-Policy/data/" )

	args = parser.parse_args()
	main(args)
