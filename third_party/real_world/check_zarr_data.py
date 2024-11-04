import zarr
import matplotlib.pyplot as plt

def read_zarr_folder(folder_path):
    try:
        zarr_data = zarr.open(folder_path, mode='r')
        print(f"Successfully opened Zarr folder: {folder_path}")
        return zarr_data
    except Exception as e:
        print(f"Error opening Zarr folder: {e}")
        return None

if __name__ == "__main__":
    folder_path = "../../3D-Diffusion-Policy/data/real-world_test_expert.zarr"
    zarr_data = read_zarr_folder(folder_path)
    if zarr_data:
        print(type(zarr_data))
        print("Contents of the Zarr group:")
        for name in zarr_data.keys():
            print(name)
        print("Zarr data shape:", zarr_data['data/wrist_img'].shape)
        import matplotlib.pyplot as plt

        # Grab the first image in wrist_img
        first_image = zarr_data['data/wrist_img'][0]
        # Inverse the blue and red channels
        first_image = first_image[..., ::-1]

        # Display the image
        plt.imshow(first_image)
        plt.title("First Image in wrist_img")
        plt.show()
        # zarr_data.tree()
        # print("Zarr data keys:", list(zarr_data.array_keys()))