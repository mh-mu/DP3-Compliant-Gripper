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
    folder_path = "../../3D-Diffusion-Policy/data/real-world_compliant30_combined16_expert.zarr"
    zarr_data = read_zarr_folder(folder_path)
    if zarr_data:
        print(type(zarr_data))
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

        # print(zarr_data['meta/episode_ends'][:])
        
        # print("Zarr data shape:", zarr_data['data/wrist_img'].shape)
        # import matplotlib.pyplot as plt

        # Get all images in 'data/wrist_img'
        images = zarr_data['data/wrist_img'][:1500]

        # Define the codec and create VideoWriter object
        height, width, layers = images[0].shape
        video = cv2.VideoWriter('wrist_img_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

        for image in images:
            # print(f"Type: {type(image)}, dtype: {image.dtype}, shape: {image.shape}")
            # Convert RGB to BGR
            # image_bgr = image[..., ::-1]
            # Convert float16 image to uint8
            image = (image * 255).astype('uint8')
            video.write(image)

        video.release()
        print("Video saved as wrist_img_video.avi")

        # # Grab the first image in wrist_img
        # first_image = zarr_data['data/wrist_img'][300]
        # print(first_image)
        # # Inverse the blue and red channels
        # first_image = first_image[..., ::-1]

        # # Display the image
        # plt.imshow(first_image)
        # plt.title("First Image in wrist_img")
        # plt.show()

        # forces = zarr_data['data/force']
        # index = np.arange(4000)

        # plt.figure(figsize=(12, 6))
        # plt.plot(index, forces[:, 0], label='X', color='r')
        # plt.plot(index, forces[:, 1], label='Y', color='g')
        # plt.plot(index, forces[:, 2], label='Z', color='b')

        # plt.xlabel('Index')
        # plt.ylabel('Force')
        # plt.legend()
        # plt.show()