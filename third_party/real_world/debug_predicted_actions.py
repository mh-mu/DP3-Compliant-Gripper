import pickle
import numpy as np
import cv2
import ast
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm
from check_zarr_data import read_zarr_folder


data_index = 8

# Load the pickle file
with open(f'rollout_data/obs_dict_list_{data_index}.pkl', 'rb') as file:
    data = pickle.load(file)


# ic(data[0]['wrist_img'].shape)
# ic(len(data))
# quit()

# Print the number of items in the pickle file
print(f"Number of items in the pickle file: {len(data)}")

# Initialize an empty list to store the actions
actions_list = []

# Load the nested lists from actions_list.txt
with open(f'rollout_data/predicted_action_list_{data_index}.pkl', 'rb') as file:
    actions_list = pickle.load(file)

# Convert the nested list into a numpy array
actions_array = np.array(actions_list)

# ic(actions_array.shape)
# quit()

# Read the Zarr folder
folder_path = "../../3D-Diffusion-Policy/data/real-world_peg_rigid_10Hz_expert.zarr"
zarr_data = read_zarr_folder(folder_path)
if zarr_data:
    training_actions = zarr_data['data/action'][:400]
else:
    raise Exception("Error reading Zarr data")


# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'rollout_data/output_{data_index}.avi', fourcc, 5.0, (1280, 480))



# Iterate through the data and actions_array
for i, (item, action_block) in tqdm(enumerate(zip(data, actions_array))):
    # Get the wrist image
    wrist_img = item['wrist_img']
    
    # Create a blank image for the 3D plot
    plot_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the last 3 elements of each size 9 array in the action block
    for row in action_block:
        ax.scatter(row[6], row[7], row[8], c='b', marker='o')
    # Plot the training actions at the training index that is 3 times i and the next two consecutive actions
    training_index = 3 * i
    for j in range(3):
        if training_index + j < len(training_actions):
            ax.scatter(training_actions[training_index + j, 6], training_actions[training_index + j, 7], training_actions[training_index + j, 8], c='purple', marker='x')
    # Set axis limits
    ax.set_xlim([-0.001, 0.001])
    ax.set_ylim([-0.001, 0.001])
    ax.set_zlim([-0.001, 0.001])
    
    # Convert the plot to an image
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Concatenate the wrist images and the plot image
    wrist_img_1 = wrist_img[0, 0]
    wrist_img_2 = wrist_img[0, 1]
    
    wrist_img_1 = np.transpose(wrist_img_1, (1, 2, 0))
    wrist_img_2 = np.transpose(wrist_img_2, (1, 2, 0))
    
    wrist_img_1 = (wrist_img_1 * 255).astype(np.uint8)
    wrist_img_2 = (wrist_img_2 * 255).astype(np.uint8)
    
    wrist_combined = np.hstack((wrist_img_1, wrist_img_2))
    combined_img = np.hstack((wrist_combined, plot_img))

    # Write the frame to the video
    out.write(combined_img)

# Release the VideoWriter object
out.release()
