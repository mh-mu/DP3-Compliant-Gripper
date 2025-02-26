import zarr
import matplotlib.pyplot as plt
import numpy as np
import cv2

def read_zarr_folder(folder_path):
    try:
        zarr_data = zarr.open(folder_path, mode='r')
        print(f"Successfully opened Zarr folder: {folder_path}")
        return zarr_data
    except Exception as e:
        print(f"Error opening Zarr folder: {e}")
        return None

if __name__ == "__main__":
    folder_path = "../../3D-Diffusion-Policy/data/real-world_contact_29_5Hz_expert.zarr"
    zarr_data = read_zarr_folder(folder_path)
    if zarr_data:
        print("Contents of the Zarr group:")
        for name in zarr_data.keys():
            print(name)

        def print_zarr_shapes(zarr_group, prefix=''):
            for key, item in zarr_group.items():
                if isinstance(item, zarr.hierarchy.Group):
                    print(f"{prefix}{key}/")
                    print_zarr_shapes(item, prefix + '  ')
                else:
                    print(f"{prefix}{key}: {item.shape}")

        print_zarr_shapes(zarr_data)

        print(zarr_data['meta/episode_ends'][:])
        
        # print("Zarr data shape:", zarr_data['data/wrist_img'].shape)

        actions = zarr_data['data/action'][:]
        # print("First 100 elements of actions:")
        # print(actions[100:200])

        # Print the range of each column in actions
        for i in range(actions.shape[1]):
            column = actions[:, i]
            print(f"Range of column {i}: min={np.min(column)}, max={np.max(column)}")

        '''
        save wrist_img as video
        '''
        images = zarr_data['data/wrist_img'][-150:]
        # images = zarr_data['data/wrist_img'][:]

        height, width, layers = images[0].shape
        video = cv2.VideoWriter('wrist_img_video_rigid_29.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))

        for image in images:
            image = (image * 255).astype('uint8')
            video.write(image)

        video.release()
        print("Video saved as wrist_img_video.avi")


        # images = zarr_data['data/gripper_img'][-600:]

        # height, width, layers = images[0].shape
        # video = cv2.VideoWriter('gripper_img_video_contact_compliant.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))

        # for image in images:
        #     image = (image * 255).astype('uint8')
        #     video.write(image)

        # video.release()
        # print("Video saved as gripper_img_video.avi")


        # images = zarr_data['data/third_view_img'][-600:]

        # height, width, layers = images[0].shape
        # video = cv2.VideoWriter('3rd_view_img_video_contact_compliant.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))

        # for image in images:
        #     image = (image * 255).astype('uint8')
        #     video.write(image)

        # video.release()
        # print("Video saved as 3rd_view_img_video.avi")

        # '''
        # plot forces
        # '''
        # num_data_points = zarr_data['data/wrist_img'].shape[0]

        # forces = zarr_data['data/force']
        # index = np.arange(num_data_points)

        # plt.figure(figsize=(12, 6))
        # plt.plot(index, forces[:, 0], label='X', color='r')
        # plt.plot(index, forces[:, 1], label='Y', color='g')
        # plt.plot(index, forces[:, 2], label='Z', color='b')

        # # index = np.arange(600)

        # # plt.figure(figsize=(12, 6))
        # # plt.plot(index, forces[:600, 0], label='X', color='r')
        # # plt.plot(index, forces[:600, 1], label='Y', color='g')
        # # plt.plot(index, forces[:600, 2], label='Z', color='b')

        # plt.xlabel('Index')
        # plt.ylabel('Force')
        # plt.legend()
        # plt.savefig('force_plot.jpg')
        # # plt.show()
    
        # # Plot the first 9900 arrays of action
        # num_data_points = min(103500, actions.shape[0])
        # # num_data_points = min(9900, actions.shape[0])
        # index = np.arange(num_data_points)

        # plt.figure(figsize=(12, 6))
        # plt.plot(index, actions[:num_data_points, 0], label='Action 4', color='r')
        # plt.plot(index, actions[:num_data_points, 1], label='Action 5', color='g')
        # plt.plot(index, actions[:num_data_points, 2], label='Action 6', color='b')

        # plt.xlabel('Index')
        # plt.ylabel('Action Value')
        # plt.legend()
        # plt.savefig('action_plot.jpg')
        # # plt.show()


    # # rewrite episode end
    # combined_dataset = zarr.open(folder_path, mode='r+')
    # # total_num_step = combined_dataset['data/action'].shape[0]
    # # combined_dataset['meta/episode_ends'] = list(range(600, total_num_step + 1, 600))
    # print(combined_dataset['meta/episode_ends'][:])