import pickle
import numpy as np
import cv2
import ast
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm


data_index = 0

# Load the pickle file
with open(f'rollout_data/obs_dict_list_{data_index}.pkl', 'rb') as file:
    data = pickle.load(file)

# Print the number of items in the pickle file
print(f"Number of items in the pickle file: {len(data)}")

# Initialize an empty list to store the actions
actions_list = []

# Load the nested lists from actions_list.txt
with open(f'rollout_data/actions_list_{data_index}.txt', 'r') as file:
    for line in file:
        # Parse each line as a separate list and append to actions_list
        actions_list.append(ast.literal_eval(line.strip()))

# Convert the nested list into a numpy array
actions_array = np.array(actions_list)

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'rollout_data/output_{data_index}.avi', fourcc, 5.0, (1280, 480))

# Iterate through the data and actions_array
for i, (item, action_block) in tqdm(enumerate(zip(data, actions_array))):
    # Get the wrist image
    wrist_img = item['wrist_img']
    
    # Extract the first action from the (3, 7) block
    action = action_block[0]
    
    # Create a blank image for the 3D plot
    plot_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all actions
    for j, act_block in enumerate(actions_array):
        act = act_block[0]
        if j == i:
            ax.scatter(act[3], act[4], act[5], c='b', marker='o')
            # Annotate the current action at a fixed position
            ax.text2D(0.05, 0.95, f'({act[3]:.6f}, {act[4]:.6f}, {act[5]:.6f})', transform=ax.transAxes, color='blue')
        else:
            ax.scatter(act[3], act[4], act[5], c='r', marker='o', alpha=0.1)
    
    # Convert the plot to an image
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Concatenate the wrist image and the plot image
    wrist_img = wrist_img[0, -1]
    wrist_img = np.transpose(wrist_img, (1, 2, 0))
    wrist_img = (wrist_img * 255).astype(np.uint8)
    combined_img = np.hstack((wrist_img, plot_img))

    # Display the combined image
    # cv2.imshow('Combined Image', combined_img)
    # cv2.waitKey(1)
    
    # Write the frame to the video
    out.write(combined_img)

# Release the VideoWriter object
out.release()