import zarr
import numpy as np
from klampt.math import so3, se3
from scipy.spatial.transform import Rotation
from icecream import ic
from tqdm import tqdm


def copy_group(source_group, target_group):
    for key, item in source_group.items():
        ic(key)
        if isinstance(item, zarr.Group):
            new_group = target_group.create_group(key)
            copy_group(item, new_group)
        else:
            if key == 'action': # next delta in trans
                processed_action, _ = calculate_new_action_state(source_group['state'][:, :12])
                target_group[key] = zarr.array(processed_action, dtype='float16')
            elif key == 'state': # prev delta in trans
                _, processed_state = calculate_new_action_state(source_group['state'][:, :12])
                target_group[key] = zarr.array(processed_state, dtype='float16')
            else:
                target_group[key] = item.astype('float16')
                ic(key)
                ic(target_group[key].shape)

def calculate_new_action_state(state):
    rotation = state[:, :9]
    translation = state[:, 9:12]

    rotation_matrices = rotation.reshape(-1, 3, 3, order='F')
    rotation_matrices = rotation_matrices.astype(np.float32)
    rotation_diff_matrices = np.array([r2 @ np.linalg.inv(r1) for r1, r2 in tqdm(zip(rotation_matrices[:-1], rotation_matrices[1:]), total=len(rotation_matrices) - 1)], dtype=np.float32)
    rotation_diff_matrices = rotation_diff_matrices[:, :-1, :]  # Remove the last row of each 3 by 3 rotation matrix
    rotation_diff = rotation_diff_matrices.reshape(-1, 6)
    
    translation_diff = translation[1:] - translation[:-1]
    
    pos_diff = np.hstack((rotation_diff, translation_diff))

    last_action = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='float32') # first two rows of identity matrix and zero translation
    action = np.vstack((pos_diff, last_action))

    first_state = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='float32') # first two rows of identity matrix and zero translation
    state = np.vstack((first_state, pos_diff))
    
    return action, state

if __name__ == "__main__":
    dataset_path =  '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_contact_expert.zarr'
    
    output_path = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_contact_processed_expert.zarr'

    new_dataset = zarr.open(output_path, mode='w')
    original_dataset = zarr.open(dataset_path, mode='r')
    copy_group(original_dataset, new_dataset)

    # Modify 'meta/episode_ends' list
    combined_dataset = zarr.open(output_path, mode='r+')
    total_num_step = combined_dataset['data/action'].shape[0]
    combined_dataset['meta/episode_ends'] = list(range(600, total_num_step + 1, 600))
    print(combined_dataset['meta/episode_ends'][:])