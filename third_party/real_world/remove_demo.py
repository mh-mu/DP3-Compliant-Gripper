import zarr

# Open the zarr dataset
zarr_path = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_contact_eval_10Hz_expert.zarr' # avoid accidental runs
# zarr_path = '/home/mh2595/project/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_contact_train_10Hz_expert.zarr'
zarr_dataset = zarr.open(zarr_path, mode='r+')

# Access the 'data' group
data_group = zarr_dataset['data']

# Iterate through subfolders in 'data' and remove the first 4800 arrays
for array_name in data_group.array_keys():
    array = data_group[array_name]
    if isinstance(array, zarr.core.Array):
        data = array[-150:]
        array.resize((150,) + array.shape[1:])
        array[:] = data
        print(f"Resized {array_name} to shape {array.shape}")
    else:
        raise TypeError(f"{array_name} is not a Zarr array or does not support resizing")
    
print(zarr_dataset['data/wrist_img'].shape)

# # Iterate through subfolders in 'data' and resize each array
# for array_name in data_group.array_keys():
#     array = data_group[array_name]
#     if isinstance(array, zarr.core.Array):
#         array.resize((array.shape[0] - 150,) + array.shape[1:])
#     else:
#         raise TypeError(f"{array_name} is not a Zarr array or does not support resizing")


# Access the 'meta' group
meta_group = zarr_dataset['meta']
print(meta_group)

# Iterate through subfolders in 'meta' and resize each array
total_num_step = zarr_dataset['data/action'].shape[0]
new_episode_ends = list(range(150, total_num_step+1, 150))
meta_group['episode_ends'].resize((len(new_episode_ends),))
meta_group['episode_ends'][:] = new_episode_ends

# new_episode_ends = [200]
# meta_group['episode_ends'].resize((len(new_episode_ends),))
# meta_group['episode_ends'][:] = new_episode_ends

# Print the new episode ends list after resizing
print("New episode ends list:", meta_group['episode_ends'][:])
