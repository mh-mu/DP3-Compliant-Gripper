import zarr


def combine_multiple_zarr_datasets(dataset_paths, output_path):
    # Create a new zarr group for the combined dataset
    combined_dataset = zarr.open(output_path, mode='w')

    def copy_group(source_group, target_group):
        for key, item in source_group.items():
            if isinstance(item, zarr.Group):
                new_group = target_group.create_group(key)
                copy_group(item, new_group)
            else:
                target_group[key] = item[:].astype('float16')

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
                    target_group[key].append(item[:].astype('float16'), axis=0)
                else:
                    target_group[key] = item[:].astype('float16')

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

# Example usage
dataset_paths = [
    '/home/mei/workspace/DP3-Compliant-Gripper/3D-Diffusion-Policy/data/real-world_compliant30_0_expert.zarr',
    '/home/mei/workspace/DP3-Compliant-Gripper/3D-Diffusion-Policy/data/real-world_compliant30_1_expert.zarr',
    '/home/mei/workspace/DP3-Compliant-Gripper/3D-Diffusion-Policy/data/real-world_compliant30_2_expert.zarr',
    '/home/mei/workspace/DP3-Compliant-Gripper/3D-Diffusion-Policy/data/real-world_compliant30_3_expert.zarr'
]
output_path = '/home/mei/workspace/DP3-Compliant-Gripper/3D-Diffusion-Policy/data/real-world_compliant30_combined16_expert.zarr'

combine_multiple_zarr_datasets(dataset_paths, output_path)

