import numpy as np
import ast
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from check_zarr_data import read_zarr_folder
import pickle
from mpl_toolkits.mplot3d import Axes3D

# Read the Zarr folder
folder_path = "../../3D-Diffusion-Policy/data/real-world_test_action_expert.zarr"
zarr_data = read_zarr_folder(folder_path)
if zarr_data:
    vr_actions = zarr_data['data/action'][:, 3:6]
    print(vr_actions.shape)
else:
    raise Exception("Error reading Zarr data")

# Load pickle file
pickle_file_path = "ur5_action_list.pkl"
with open(pickle_file_path, 'rb') as f:
    ur5_actions = np.array(pickle.load(f))

# Print shape of the content
print(ur5_actions.shape)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

index = 0

# Set axis limits
x_limits = [min(np.min(vr_actions[:, 0]), np.min(ur5_actions[:, 0])), max(np.max(vr_actions[:, 0]), np.max(ur5_actions[:, 0]))]
y_limits = [min(np.min(vr_actions[:, 1]), np.min(ur5_actions[:, 1])), max(np.max(vr_actions[:, 1]), np.max(ur5_actions[:, 1]))]
z_limits = [min(np.min(vr_actions[:, 2]), np.min(ur5_actions[:, 2])), max(np.max(vr_actions[:, 2]), np.max(ur5_actions[:, 2]))]

def plot_pair(index):
    ax.clear()
    if index + 1 < len(ur5_actions):
        # Plot VR action vector
        ax.quiver(0, 0, 0, vr_actions[index, 0], vr_actions[index, 1], vr_actions[index, 2], color='b', label='VR Action')
        # Plot UR5 action vector
        ax.quiver(0, 0, 0, ur5_actions[index + 1, 0], ur5_actions[index + 1, 1], ur5_actions[index + 1, 2], color='r', label='UR5 Action')
        
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        
        # Set axis limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        
        # Display the values of the pairs
        print(f"VR Action: {vr_actions[index]}, UR5 Action: {ur5_actions[index + 1]}")
        
        # Add legend
        ax.legend()
        
        plt.draw()

def on_key(event):
    global index
    if event.key == 'right':
        index = (index + 1) % len(vr_actions)
    elif event.key == 'left':
        index = (index - 1) % len(vr_actions)
    plot_pair(index)

fig.canvas.mpl_connect('key_press_event', on_key)
plot_pair(index)
plt.show()