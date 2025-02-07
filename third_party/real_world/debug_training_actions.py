from check_zarr_data import read_zarr_folder
import math
import sys, os
import time
from icecream import ic
from tqdm import tqdm
from klampt.math import so3, se3
from mpl_toolkits.mplot3d import Axes3D
import cv2
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UR5_IMPEDANCE')))

from ur5_controller_wrapper import ur5ControlWrapper


UR5_ip = '192.168.0.101'
HOME_t_obj = [-0.6, 0, 0.08]
R_EE_WORLD_HOME = [0,0,-1, math.sqrt(2)/2,math.sqrt(2)/2,0,math.sqrt(2)/2,-math.sqrt(2)/2,0] #klampt format
R = [-0.036093246883230824, -0.0016417553230183065, -0.9993470779308286, 0.8343056200142569, 0.5504278824056338, -0.03103673119186873, 0.5501194506123673, -0.8348810998638722, -0.018497003758397978]
t = [-0.5719669638367115, -0.13510426694812794, 0.00818264088493828]
UR5_home_position = (R, t)

if __name__ == "__main__":
    folder_path = "../../3D-Diffusion-Policy/data/real-world_peg_rigid_10Hz_expert.zarr"
    zarr_data = read_zarr_folder(folder_path)
    if zarr_data:
        actions = zarr_data['data/action'][:400]
        images = zarr_data['data/wrist_img'][:400]

        # get trans actions
        trans_actions = actions[:, 6:9]

        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('debug_training_action.avi', fourcc, 10.0, (1280, 480))

        # Determine the bounds for the action plot
        x_min, x_max = np.min(trans_actions[:, 0]) / 5, np.max(trans_actions[:, 0]) / 2
        y_min, y_max = np.min(trans_actions[:, 1]), np.max(trans_actions[:, 1]) / 10
        z_min, z_max = np.min(trans_actions[:, 2]), np.max(trans_actions[:, 2]) / 10

        for i in tqdm(range(len(images))):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot all actions
            ax.scatter(trans_actions[:, 0], trans_actions[:, 1], trans_actions[:, 2], label='All Actions', alpha=0.1)

            # Highlight the action at the current index
            ax.scatter(trans_actions[i, 0], trans_actions[i, 1], trans_actions[i, 2], color='r', label='Current Action')

            # Print the values of the plotted action in the plot
            ax.text2D(0.05, 1.00, 
                      f'Current Action: ({trans_actions[i, 0]:.6f}, {trans_actions[i, 1]:.6f}, {trans_actions[i, 2]:.6f})', 
                      transform=ax.transAxes, color='red')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

            # Set the axis limits to zoom in on the region where there are actions
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            # Save the plot as an image
            plt.savefig('actions_plot.png')
            plt.close(fig)

            # Read the action plot image
            action_img = cv2.imread('actions_plot.png')

            wrist_img = (images[i] * 255).astype(np.uint8)

            # Resize the action plot image to match the wrist image height
            action_img = cv2.resize(action_img, (640, 480))

            # Concatenate the action plot and wrist image side by side
            combined_img = cv2.hconcat([wrist_img, action_img])

            # Write the combined image to the video
            out.write(combined_img)

        # Release the video writer
        out.release()

        # ur5_controller = ur5ControlWrapper(home_T = (R_EE_WORLD_HOME, HOME_t_obj) , ip = UR5_ip, ft_sensor=None)
        # time.sleep(2)

        # ur5_controller.set_EE_transform_linear(UR5_home_position, max_trans_v = 0.8)
        # time.sleep(1)

        # for action in tqdm(actions):
        #     rot_vec = action[:3]
        #     rot = so3.from_rotation_vector(rot_vec)
        #     trans = action[3:6].tolist()

        #     ur5_controller.set_EE_transform_delta((rot, trans))
        #     time.sleep(0.1)