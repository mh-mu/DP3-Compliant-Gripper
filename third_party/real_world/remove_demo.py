import zarr

# Open the zarr dataset
zarr_path = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real-world_circle_expert.zarr'
zarr_dataset = zarr.open(zarr_path, mode='r+')

# # Access the 'data' group
# data_group = zarr_dataset['data']

# # Iterate through subfolders in 'data' and resize each array
# for array_name in data_group.array_keys():
#     array = data_group[array_name]
#     if isinstance(array, zarr.core.Array):
#         array.resize((array.shape[0] - 300,) + array.shape[1:])
#     else:
#         raise TypeError(f"{array_name} is not a Zarr array or does not support resizing")


# Access the 'meta' group
meta_group = zarr_dataset['meta']

# Iterate through subfolders in 'meta' and resize each array
new_episode_ends = list(range(300, 5401, 300))
meta_group['episode_ends'].resize((len(new_episode_ends),))
meta_group['episode_ends'][:] = new_episode_ends
