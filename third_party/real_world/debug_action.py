import numpy as np
import ast
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from check_zarr_data import read_zarr_folder

# Read the Zarr folder
folder_path = "../../3D-Diffusion-Policy/data/real-world_line_expert.zarr"
zarr_data = read_zarr_folder(folder_path)
if zarr_data:
    training_actions = zarr_data['data/action'][:]
else:
    raise Exception("Error reading Zarr data")

# Initialize an empty list to store the actions
actions_list = []

# Load the nested lists from actions_list.txt
with open('actions_list.txt', 'r') as file:
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

# Plot 3D plot of the two arrays
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot training actions
ax.scatter(training_actions_6d[:, 0], training_actions_6d[:, 1], training_actions_6d[:, 2], alpha=0.02, label='Training Actions', c='blue')

# Plot predicted actions
ax.scatter(predicted_actions_6d[:, 0], predicted_actions_6d[:, 1], predicted_actions_6d[:, 2], alpha=0.5, label='Predicted Actions', c='red')

ax.set_title('3D Plot of Training and Predicted Actions (trans)')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.legend()
plt.show()

# # Reshape the array to 2D for t-SNE
# predicted_actions_6d = actions_array_6d.reshape(-1, actions_array_6d.shape[2])

# # Perform t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(predicted_actions_6d)

# # Perform t-SNE on training actions
# tsne_training = TSNE(n_components=2, random_state=42)
# tsne_training_results = tsne_training.fit_transform(training_actions_6d)

# # Plot both the training actions and predicted actions
# plt.figure(figsize=(10, 8))
# plt.scatter(tsne_training_results[:, 0], tsne_training_results[:, 1], alpha=0.5, label='Training Actions', c='blue')
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5, label='Predicted Actions', c='red')
# plt.title('t-SNE Visualization of Training and Predicted Actions (First 6 Dimensions)')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.legend()
# plt.savefig('tsne_plot_combined_6d.jpg', format='jpg')
