# bash scripts/metaworld/gen_demonstration_expert.sh reach 5
import argparse
import os
import zarr
import numpy as np
from diffusion_policy_3d.env.real_world import RealWorldEnv
from termcolor import cprint
import copy
import imageio
import cv2
# from metaworld.policies import *
# import faulthandler
# faulthandler.enable()
import sys, time
import openvr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'UR5_IMPEDANCE')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'UR5_Teleop')))

import pyspacemouse
from vive_controller_teleop import *
from scipy.spatial.transform import Rotation
from klampt.math import so3, se3
from diffusion_policy_3d.env.real_world import CONSTANTS
import keyboard
from icecream import ic 

seed = np.random.randint(0, 100)

def main(args):
	env_name = args.env_name
	demo_device = args.demo_device
	
	save_dir = os.path.join(args.root_dir, 'real-world_'+args.env_name+'_expert.zarr')
	if os.path.exists(save_dir):
		cprint('Data already exists at {}'.format(save_dir), 'red')
		cprint("If you want to overwrite, delete the existing directory first.", "red")
		cprint("Do you want to overwrite? (y/n)", "red")
		user_input = 'y'
		if user_input == 'y':
			cprint('Overwriting {}'.format(save_dir), 'red')
			os.system('rm -rf {}'.format(save_dir))
		else:
			cprint('Exiting', 'red')
			return
	os.makedirs(save_dir, exist_ok=True)

	e = RealWorldEnv(env_name, demo_device, device="cuda:0")
	
	num_episodes = args.num_episodes
	cprint(f"Number of episodes : {num_episodes}", "yellow")

	total_count = 0
	wrist_img_arrays = []
	force_arrays = []
	state_arrays = []
	action_arrays = []
	episode_ends_arrays = []
	
	episode_idx = 0

	if demo_device == 'spacemouse':
		if_spacemouse_success = pyspacemouse.open()
	elif demo_device == 'vr':
		init_openvr()
		init_controllers()
		time.sleep(1)
		print("Vive Ready")
		prev_vr_pose = get_controller_pose()
	
	# loop over episodes
	while episode_idx < num_episodes:
		# cprint(f'Press any key to begin episode {episode_idx}', 'yellow')
		# keyboard.wait()

		e.reset()
		prev_gripper_action = CONSTANTS.OPEN

		obs_dict = e.get_visual_obs()

		done = False

		wrist_img_arrays_sub = []
		force_arrays_sub = []
		state_arrays_sub = []
		action_arrays_sub = []
		total_count_sub = 0
  
		while not done:
			total_count_sub += 1
			
			obs_wrist_img = obs_dict['wrist_img']
			obs_force = obs_dict['force']
			obs_robot_state = obs_dict['state']

			wrist_img_arrays_sub.append(obs_wrist_img)
			force_arrays_sub.append(obs_force)
			state_arrays_sub.append(obs_robot_state)
			
			if demo_device == 'spacemouse':
				if if_spacemouse_success:
					spacemouse_state = pyspacemouse.read()
					trans = np.array([spacemouse_state.y, -spacemouse_state.x, spacemouse_state.z]) * e.trans_scale
					rot_rad = np.array([spacemouse_state.roll, spacemouse_state.pitch, -spacemouse_state.yaw]) * e.rot_scale
					rot_vec = Rotation.from_euler('xyz', rot_rad, degrees=True).as_rotvec()

					gripper_action = prev_gripper_action
					if spacemouse_state.buttons[0] == 1:
						gripper_action = CONSTANTS.OPEN
					elif spacemouse_state.buttons[1] == 1:
						gripper_action = CONSTANTS.CLOSE
					prev_gripper_action = gripper_action

					action = np.concatenate((rot_vec, trans, [gripper_action]))
				else:
					cprint(f'Error: Spacemouse not reading', 'red')
					action = np.zeros(7)
			elif demo_device == 'vr':
				vr_pose = get_controller_pose()
				delta_T = get_controller_pose_delta(vr_pose, prev_vr_pose)
				delta_rot_vec = Rotation.from_matrix(delta_T[:3, :3]).as_rotvec() * e.rot_scale
				delta_trans = delta_T[:3, 3] * e.trans_scale

				# TODO: track gripper action
				gripper_action = CONSTANTS.OPEN

				prev_vr_pose = vr_pose
				if is_trigger_active():
					action = np.concatenate((delta_rot_vec, delta_trans, [gripper_action]))
				else:
					action = np.zeros(7) # TODO: gripper action not right
		
			action_arrays_sub.append(action)
			obs_dict, _, done, _ = e.step(action)
   
			if done:
				break

		total_count += total_count_sub
		episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    
		wrist_img_arrays.extend(copy.deepcopy(wrist_img_arrays_sub))
		force_arrays.extend(copy.deepcopy(force_arrays_sub))
		state_arrays.extend(copy.deepcopy(state_arrays_sub))
		action_arrays.extend(copy.deepcopy(action_arrays_sub))
		ic(len(action_arrays))
		cprint('Episode: {}'.format(episode_idx), 'green')
		episode_idx += 1

	# e.cap.release() # debug
	openvr.shutdown()
	e.ur5_controller.close()

 	###############################
    # save data
    ###############################
    # create zarr file
	zarr_root = zarr.group(save_dir)
	zarr_data = zarr_root.create_group('data')
	zarr_meta = zarr_root.create_group('meta')
	# save img, state, action arrays into data, and episode ends arrays into meta
	wrist_img_arrays = np.stack(wrist_img_arrays, axis=0)
	if wrist_img_arrays.shape[1] == 3: # make channel last
		wrist_img_arrays = np.transpose(wrist_img_arrays, (0,2,3,1))
	force_arrays = np.stack(force_arrays, axis=0)
	state_arrays = np.stack(state_arrays, axis=0)
	action_arrays = np.stack(action_arrays, axis=0)
	episode_ends_arrays = np.array(episode_ends_arrays)

	compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
	wrist_img_chunk_size = (100, wrist_img_arrays.shape[1], wrist_img_arrays.shape[2], wrist_img_arrays.shape[3])
	force_chunk_size = (100, force_arrays.shape[1])
	state_chunk_size = (100, state_arrays.shape[1])
	action_chunk_size = (100, action_arrays.shape[1])
	zarr_data.create_dataset('wrist_img', data=wrist_img_arrays, chunks=wrist_img_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('force', data=force_arrays, chunks=force_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

	cprint(f'-'*50, 'cyan')
	# print shape
	cprint(f'wrist img shape: {wrist_img_arrays.shape}, range: [{np.min(wrist_img_arrays)}, {np.max(wrist_img_arrays)}]', 'green')
	cprint(f'force shape: {force_arrays.shape}, range: [{np.min(force_arrays)}, {np.max(force_arrays)}]', 'green')
	cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
	cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
	cprint(f'Saved zarr file to {save_dir}', 'green')

	# clean up
	del wrist_img_arrays, force_arrays, state_arrays, action_arrays, episode_ends_arrays
	del zarr_root, zarr_data, zarr_meta
	del e


 
if __name__ == "__main__":
    
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='test')
	parser.add_argument('--demo_device', type=str, default='vr')
	parser.add_argument('--num_episodes', type=int, default=10)
	parser.add_argument('--root_dir', type=str, default="../../3D-Diffusion-Policy/data/" )

	args = parser.parse_args()
	main(args)
