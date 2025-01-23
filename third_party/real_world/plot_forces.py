import pickle
import os
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
    
folder_path = "../../3D-Diffusion-Policy/data/real-world_contact_compliant_10Hz_expert.zarr"
zarr_data = read_zarr_folder(folder_path)
    

# Prompt the user to input a list of paths for the pickle files
pickle_files = ['/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/third_party/real_world/rollout_data/force_list_compliant_withForce.pkl',
                '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/third_party/real_world/rollout_data/force_list.pkl']

# Initialize lists to store the forces
all_x_forces = []
all_y_forces = []
all_z_forces = []

# Load and process each pickle file
for pickle_file_path in pickle_files:
    pickle_file_path = pickle_file_path.strip()
    with open(pickle_file_path, 'rb') as f:
        forces = pickle.load(f)
    
    # Check if the loaded data is a list of (3,) forces
    if not all(len(force) == 3 for force in forces):
        raise ValueError(f"The loaded data from {pickle_file_path} is not a list of (3,) forces")
    
    # Extract the x, y, z components of the forces
    x_forces = [force[0] for force in forces]
    y_forces = [force[1] for force in forces]
    z_forces = [force[2] for force in forces]
    
    all_x_forces.append(x_forces)
    all_y_forces.append(y_forces)
    all_z_forces.append(z_forces)

# Plot the forces from all files and the zarr data
plt.figure(figsize=(10, 6))

# Plot Zarr data forces
if zarr_data:
    zarr_forces = zarr_data['data/force']
    num_points = zarr_forces.shape[0]
    segment_length = 150

    # Plot the 900-1050 points from zarr_forces
    zarr_z_forces_segment = zarr_forces[450:600, 2]
    plt.plot(zarr_z_forces_segment, label='Example Training Force', linestyle='--', color='green', alpha=0.7)
    
    # for start in range(0, num_points, segment_length):
    #     end = min(start + segment_length, num_points)
    #     zarr_z_forces = zarr_forces[start:end, 2]
    #     # plt.plot(zarr_z_forces, label=f'Demo Z Force {start//segment_length + 1}', linestyle='--', color='green', alpha=0.1)
    #     plt.plot(zarr_z_forces, linestyle='--', color='green', alpha=0.1)

# Plot pickle file forces
for i, z_forces in enumerate(all_z_forces):
    # plt.plot(z_forces, label=f'Rollout Z Force {i+1}')
    if i == 0:
        plt.plot(z_forces, label='with force')
    elif i == 1:
        plt.plot(z_forces, label='no force')

plt.xlabel('Time Step')
plt.ylabel('Force')
plt.legend()
plt.grid(True)
plt.show()
