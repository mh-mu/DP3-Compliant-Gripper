import zarr

# Open the zarr dataset
zarr_path = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_peg_rigid_single_10Hz_expert.zarr' # avoid accidental runs
zarr_dataset = zarr.open(zarr_path, mode='r+')

# Access the 'data' group
data_group = zarr_dataset['data']
# Iterate through subfolders in 'data' and remove the last 600 arrays
for array_name in data_group.array_keys():
    array = data_group[array_name]
    if isinstance(array, zarr.core.Array):
        # array.resize((array.shape[0] - 600,) + array.shape[1:])
        array.resize((200,) + array.shape[1:])
        print(f"Resized {array_name} to shape {array.shape}")
    else:
        raise TypeError(f"{array_name} is not a Zarr array or does not support resizing")
    
print(zarr_dataset['data/wrist_img'].shape)

# # Iterate through subfolders in 'data' and resize each array
# for array_name in data_group.array_keys():
#     array = data_group[array_name]
#     if isinstance(array, zarr.core.Array):
#         array.resize((array.shape[0] - 300,) + array.shape[1:])
#     else:
#         raise TypeError(f"{array_name} is not a Zarr array or does not support resizing")


# Access the 'meta' group
meta_group = zarr_dataset['meta']

# # Iterate through subfolders in 'meta' and resize each array
# new_episode_ends = list(range(450, 9901, 450))
# meta_group['episode_ends'].resize((len(new_episode_ends),))
# meta_group['episode_ends'][:] = new_episode_ends

new_episode_ends = [200]
meta_group['episode_ends'].resize((len(new_episode_ends),))
meta_group['episode_ends'][:] = new_episode_ends

# Print the new episode ends list after resizing
print("New episode ends list:", meta_group['episode_ends'][:])
