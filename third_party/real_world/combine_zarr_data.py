import zarr
from realworld_utils import *
from icecream import ic
from scipy.spatial.transform import Rotation as R

def sum_skipped_data(data, stepskip):
    rot_vecs = data[:, :6]
    rot_mats = rotation_6d_to_matrix_batch(rot_vecs)
    trans = data[:, 6:9]

    combined_rot_mats = []
    combined_trans = []

    num_steps = rot_mats.shape[0]

    for i in range(0, num_steps, stepskip):
        combined_rot_mat = np.eye(3)
        for j in range(stepskip):
            if i + j < num_steps:
                combined_rot_mat = rot_mats[i + j] @ combined_rot_mat
        combined_rot_mats.append(combined_rot_mat[:2].flatten())
        combined_trans.append(trans[i:i + stepskip].sum(axis=0))

    combined_data = np.hstack((np.array(combined_rot_mats), np.array(combined_trans)))
    return combined_data.astype('float16')


def combine_multiple_zarr_datasets(dataset_paths, output_path, stepskip=3):
    # Create a new zarr group for the combined dataset
    combined_dataset = zarr.open(output_path, mode='w')

    def copy_group(source_group, target_group):
        for key, item in source_group.items():
            ic(key)
            if isinstance(item, zarr.Group):
                new_group = target_group.create_group(key)
                copy_group(item, new_group)
            else:
                if key == 'action':
                    target_group[key] = sum_skipped_data(item, stepskip)
                elif key == 'state':
                    target_group[key] = sum_skipped_data(item, stepskip)
                else:
                    target_group[key] = item[::stepskip].astype('float16')

    def append_group(source_group, target_group):
        for key, item in source_group.items():
            if isinstance(item, zarr.Group):
                if key not in target_group:
                    new_group = target_group.create_group(key)
                else:
                    new_group = target_group[key]
                append_group(item, new_group)
            else:
                if key in target_group:
                    if key == 'action':
                        # Sum the actions when skipping
                        new_action = sum_skipped_data(item, stepskip)
                        target_group[key].append(new_action, axis=0)
                    elif key == 'state':
                        new_state = sum_skipped_data(item, stepskip)
                        target_group[key].append(new_state, axis=0)
                    else:
                        target_group[key].append(item[::stepskip].astype('float16'), axis=0)
                else:
                    if key == 'action':
                        target_group[key] = sum_skipped_data(item, stepskip)
                    elif key == 'state':
                        target_group[key] = sum_skipped_data(item, stepskip)
                    else:
                        target_group[key] = item[::stepskip].astype('float16')

    # Copy the first dataset into the combined dataset
    first_dataset = zarr.open(dataset_paths[0], mode='r')
    copy_group(first_dataset, combined_dataset)
    print('Copied first dataset')

    # Append the rest of the datasets into the combined dataset
    for dataset_path in dataset_paths[1:]:
        print('Appending dataset:', dataset_path)
        dataset = zarr.open(dataset_path, mode='r')
        append_group(dataset, combined_dataset)

    return combined_dataset

if __name__ == "__main__":
    dataset_paths = [
        # '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_peg_rigid_processed_expert.zarr'
        '/home/mh2595/project/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_contact_29_10Hz_expert.zarr'
    ]
    # output_path = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_peg_rigid_10Hz_expert.zarr'
    output_path = '/home/mh2595/project/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_contact_29_5Hz_expert.zarr'

    combine_multiple_zarr_datasets(dataset_paths, output_path, stepskip=2)

    # # Open the original dataset
    # original_dataset = zarr.open(dataset_paths[0], mode='r')
    # original_actions = original_dataset['data/action'][90:120]

    # # Convert the first 6 elements back to rotation matrices
    # rot_vecs = original_actions[:, :6]
    # rot_mats = rotation_6d_to_matrix_batch(rot_vecs)

    # # Convert rotation matrices to Euler angles
    # euler_angles = R.from_matrix(rot_mats).as_euler('xyz', degrees=True)

    # # Print the Euler angles
    # print(euler_angles)

    # # Open the combined dataset
    # combined_dataset = zarr.open(output_path, mode='r')
    # combined_actions = combined_dataset['data/action'][30:40]

    # # Convert the first 6 elements back to rotation matrices
    # combined_rot_vecs = combined_actions[:, :6]
    # combined_rot_mats = rotation_6d_to_matrix_batch(combined_rot_vecs)

    # # Convert rotation matrices to Euler angles
    # combined_euler_angles = R.from_matrix(combined_rot_mats).as_euler('xyz', degrees=True)

    # # Print the Euler angles
    # print(combined_euler_angles)

    # Modify 'meta/episode_ends' list
    combined_dataset = zarr.open(output_path, mode='r+')
    total_num_step = combined_dataset['data/action'].shape[0]
    combined_dataset['meta/episode_ends'] = list(range(75, total_num_step + 1, 75))
    print(combined_dataset['meta/episode_ends'][:])
