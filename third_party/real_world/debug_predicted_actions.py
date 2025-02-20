import pickle
import numpy as np
import cv2
import ast
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm
from check_zarr_data import read_zarr_folder


data_index = 21

with open(f'rollout_data/obs_dict_list_{data_index}.pkl', 'rb') as file:
    data = pickle.load(file)

print(f"Number of items in the pickle file: {len(data)}")

actions_list = []
with open(f'rollout_data/predicted_action_list_{data_index}.pkl', 'rb') as file:
    actions_list = pickle.load(file)
actions_array = np.array(actions_list)


folder_path = "../../3D-Diffusion-Policy/data/real-world_peg_eval_10Hz_expert.zarr"
zarr_data = read_zarr_folder(folder_path)
if zarr_data:
    training_actions = zarr_data['data/action'][:200]
else:
    raise Exception("Error reading Zarr data")

fourcc = cv2.VideoWriter_fourcc(*'VP80')
out = cv2.VideoWriter(f'rollout_data/output_{data_index}.webm', fourcc, 5.0, (640*3, 480))


all_percentage_diffs = []

for i, (item, action_block) in tqdm(enumerate(zip(data, actions_array))):
    wrist_img = item['wrist_img']
    
    plot_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the last 3 elements of each size 9 array in the action block
    for idx, row in enumerate(action_block):
        ax.scatter(row[6], row[7], row[8], c='blue', marker='o')
    # Plot the training actions at the training index that is 3 times i and the next two consecutive actions
    training_index = 0 if i == 0 else 3 * i - 1
    for j in range(3):
        if training_index + j < len(training_actions):
            ax.scatter(training_actions[training_index + j, 6], training_actions[training_index + j, 7], training_actions[training_index + j, 8], c='purple', marker='x')
    
    # Calculate and display the percentage difference
    percentage_diffs = []
    for j in range(3):
        if training_index + j < len(training_actions):
            pred = action_block[j][6:9]
            gt = training_actions[training_index + j, 6:9]
            diff = np.abs(pred - gt) / np.abs(gt) * 100
            percentage_diffs.append(diff)
    all_percentage_diffs.extend(percentage_diffs)
    
    # Add text on top of the plot
    pred_text_str = "\n".join([f"Pred: ({row[6]:.8f}, {row[7]:.8f}, {row[8]:.8f})" for row in action_block])
    train_text_str = "\n".join([f"Train: ({training_actions[training_index + j, 6]:.8f}, {training_actions[training_index + j, 7]:.8f}, {training_actions[training_index + j, 8]:.8f})" 
                                for j in range(3) if training_index + j < len(training_actions)])
    diff_text_str = "\n".join([f"Diff {j+1}: ({diff[0]:.2f}%, {diff[1]:.2f}%, {diff[2]:.2f}%)" for j, diff in enumerate(percentage_diffs)])
    
    plt.figtext(0.1, 0.9, pred_text_str, wrap=True, horizontalalignment='left', fontsize=8, color='blue')
    plt.figtext(0.1, 0.8, train_text_str, wrap=True, horizontalalignment='left', fontsize=8, color='purple')
    plt.figtext(0.1, 0.7, diff_text_str, wrap=True, horizontalalignment='left', fontsize=8, color='red')
    # Set axis limits
    ax.set_xlim([-0.0015, 0.0015])
    ax.set_ylim([-0.0015, 0.0015])
    ax.set_zlim([-0.0015, 0.0015])
    
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

    # # Display the combined image
    # cv2.imshow('Combined Image', combined_img)
    
    # Wait for a key press to proceed to the next frame
    key = cv2.waitKey(1)

    # Write the frame to the video
    out.write(combined_img)

# Release the VideoWriter object
out.release()

# Save all percentage differences to a file
with open(f'rollout_data/percentage_diffs_{data_index}.pkl', 'wb') as file:
    pickle.dump(all_percentage_diffs, file)

# Calculate and display the average percentage difference
average_diff = np.mean(all_percentage_diffs, axis=0)
print(f"Average percentage difference: {average_diff}")
