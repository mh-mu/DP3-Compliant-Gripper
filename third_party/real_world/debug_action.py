import numpy as np
import ast
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from check_zarr_data import read_zarr_folder

# Read the Zarr folder
folder_path = "../../3D-Diffusion-Policy/data/real-world_line_10Hz_expert.zarr"
zarr_data = read_zarr_folder(folder_path)
if zarr_data:
    training_actions = zarr_data['data/action'][:]
else:
    raise Exception("Error reading Zarr data")

# folder_path2 = "../../3D-Diffusion-Policy/data/real-world_line_expert.zarr"
# zarr_data2 = read_zarr_folder(folder_path2)
# if zarr_data2:
#     training_actions2 = zarr_data2['data/action'][:]
# else:
#     raise Exception("Error reading Zarr data")

# Initialize an empty list to store the actions
actions_list = []

# Load the nested lists from actions_list.txt
with open('rollout_data/actions_list_23.txt', 'r') as file:
    for line in file:
        # Parse each line as a separate list and append to actions_list
        actions_list.append(ast.literal_eval(line.strip()))

# Convert the nested list into a numpy array
actions_array = np.array(actions_list)

# Print the shape of the array
print("Shape of the array:", actions_array.shape)

# Calculate and print the max and min values along the third dimension (the one with 7)
for i in range(actions_array.shape[2]):
    slice_ = actions_array[:, :, i]
    slice_max = slice_.max()
    slice_min = slice_.min()
    print(f"Slice {i} - Min: {slice_min}, Max: {slice_max}")

# Consider only the first 6 dimensions out of the 7 dimensions
actions_array_6d = actions_array[:, :, 3:6]
# actions_array_6d = actions_array[:, :, :3]
predicted_actions_6d = actions_array_6d.reshape(-1, actions_array_6d.shape[2])
training_actions_6d = training_actions[:, 3:6]
# training_actions_6d = training_actions[:, :3]

# training_actions_3d = training_actions[:, 3:6]
# training_actions2_3d = training_actions2[:, 3:6]

# # Calculate and print the average of both arrays along the columns
# average_training_actions_3d = np.mean(training_actions_3d, axis=0)
# average_training_actions2_3d = np.mean(training_actions2_3d, axis=0)

# print("Average of training_actions_3d along the columns:", average_training_actions_3d)
# print("Average of training_actions2_3d along the columns:", average_training_actions2_3d)

# Plot 3D plot of the two arrays
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot training actions
ax.scatter(training_actions_6d[:, 0], training_actions_6d[:, 1], training_actions_6d[:, 2], alpha=0.02, label='Training Actions', c='blue')
# ax.scatter(training_actions_3d[:, 0], training_actions_3d[:, 1], training_actions_3d[:, 2], alpha=0.02, label='Training Actions', c='blue')

# Plot predicted actions
ax.scatter(predicted_actions_6d[:, 0], predicted_actions_6d[:, 1], predicted_actions_6d[:, 2], alpha=0.5, label='Predicted Actions', c='red')
# ax.scatter(training_actions2_3d[:, 0], training_actions2_3d[:, 1], training_actions2_3d[:, 2], alpha=0.02, label='Training Actions', c='red')

ax.set_title('3D Plot of Training and Predicted Actions (trans)')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.legend()
plt.show()

