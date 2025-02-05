import zarr
import numpy as np
from klampt.math import so3, se3
from scipy.spatial.transform import Rotation
from icecream import ic

def copy_group(source_group, target_group):
    for key, item in source_group.items():
        if isinstance(item, zarr.Group):
            new_group = target_group.create_group(key)
            copy_group(item, new_group)
        else:
            if key == 'action': # next action
                action = calculate_new_action_state(source_group['state'][:, :12])
                target_group[key] = zarr.array(action, dtype='float16')
            elif key == 'state': # prev action
                new_state = calculate_state(source_group[key][:])
                target_group[key] = zarr.array(new_state, dtype='float16')
            else:
                target_group[key] = item.astype('float16')
                ic(key)
                ic(target_group[key].shape)

def calculate_new_action_state(state):
    rotation = state[:, :9]
    translation = state[:, 9:12]

    rotation_matrices = rotation.reshape(-1, 3, 3, order='F')
    rotation_vectors = np.array([Rotation.from_matrix(rot).as_rotvec() for rot in rotation_matrices])
    
    rotation_diff = rotation_vectors[1:] - rotation_vectors[:-1]
    translation_diff = translation[1:] - translation[:-1]
    
    action = np.hstack((rotation_diff, translation_diff))
    last_action = np.zeros((1, 6), dtype='float16') 
    action = np.vstack((action, last_action))
    
    return action

def calculate_state(state):
    rotation = state[:, :9]
    translation = state[:, 9:12]

    rotation_matrices = rotation.reshape(-1, 3, 3, order='F')
    rotation_vectors = np.array([Rotation.from_matrix(rot).as_rotvec() for rot in rotation_matrices])
    
    rotation_diff_vec = np.vstack((np.zeros((1, 3)), rotation_vectors[1:] - rotation_vectors[:-1]))
    rotation_diff_mat = np.array([Rotation.from_rotvec(rot).as_matrix() for rot in rotation_diff_vec])
    rotation_diff = rotation_diff_mat[:, :, :2].reshape(-1, 6)
    translation_diff = np.vstack((np.zeros((1, 3)), translation[1:] - translation[:-1]))
    
    new_state = np.hstack((rotation_diff, translation_diff))
    
    return new_state

if __name__ == "__main__":
    dataset_path =  '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_peg_rigid_expert.zarr'
    
    output_path = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_peg_rigid_processed_expert.zarr'

    new_dataset = zarr.open(output_path, mode='w')
    # Copy the first dataset into the combined dataset
    original_dataset = zarr.open(dataset_path, mode='r')
    copy_group(original_dataset, new_dataset)

    # Modify 'meta/episode_ends' list
    combined_dataset = zarr.open(output_path, mode='r+')
    total_num_step = combined_dataset['data/action'].shape[0]
    combined_dataset['meta/episode_ends'] = list(range(600, total_num_step + 1, 600))
    print(combined_dataset['meta/episode_ends'][:])