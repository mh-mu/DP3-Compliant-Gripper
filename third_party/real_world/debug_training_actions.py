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
    folder_path = "../../3D-Diffusion-Policy/data/real-world_line_expert.zarr"
    zarr_data = read_zarr_folder(folder_path)
    if zarr_data:
        actions = zarr_data['data/action'][:300]
        images = zarr_data['data/wrist_img'][:300]

        # get trans actions
        actions = actions[:, 3:6]

        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))

        for i in tqdm(range(len(images))):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot all actions
            ax.scatter(actions[:, 0], actions[:, 1], actions[:, 2], label='All Actions', alpha=0.02)

            # Highlight the action at the current index
            ax.scatter(actions[i, 0], actions[i, 1], actions[i, 2], color='r', label='Current Action')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

            # Save the plot as an image
            plt.savefig('actions_plot.png')
            plt.close(fig)

            # Read the action plot image
            action_img = cv2.imread('actions_plot.png')

            wrist_img = (images[i] * 255).astype(np.uint8)

            # Concatenate the action plot and wrist image side by side
            combined_img = cv2.hconcat([action_img, wrist_img])
            # Display the combined image
            cv2.imshow('Combined Image', combined_img)
            cv2.waitKey(1)

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