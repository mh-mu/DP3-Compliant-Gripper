import zarr
from realworld_utils import *
from icecream import ic

def combine_multiple_zarr_datasets(dataset_paths, output_path):
    # Create a new zarr group for the combined dataset
    combined_dataset = zarr.open(output_path, mode='w')

    def copy_group(source_group, target_group):
        for key, item in source_group.items():
            ic(key)
            if isinstance(item, zarr.Group):
                new_group = target_group.create_group(key)
                copy_group(item, new_group)
            else:
                target_group[key] = item[:]

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
                    target_group[key].append(item[:], axis=0)
                else:
                    target_group[key] = item[:]

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
        '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_hover0_expert.zarr',
        '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_hover1_expert.zarr',
        '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_hover2_expert.zarr',
        '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_hover3_expert.zarr',
        '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_hover4_expert.zarr',
        '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_hover5_expert.zarr',
    ]
    output_path = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_hover_expert.zarr'
    # output_path = '/home/mh2595/project/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_contact_eval_5Hz_expert.zarr'

    combine_multiple_zarr_datasets(dataset_paths, output_path)

    # Modify 'meta/episode_ends' list
    combined_dataset = zarr.open(output_path, mode='r+')
    total_num_step = combined_dataset['data/action'].shape[0]
    combined_dataset['meta/episode_ends'] = list(range(300, total_num_step + 1, 300))
    print(combined_dataset['meta/episode_ends'][:])
