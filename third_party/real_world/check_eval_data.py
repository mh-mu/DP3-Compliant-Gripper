import pickle
from PIL import Image
import numpy as np
import cv2

# Load the pickle file
with open('obs_dict_list.pkl', 'rb') as file:
    data = pickle.load(file)

# Print the number of items in the pickle file
print(f"Number of items in the pickle file: {len(data)}")

# Print the shape of the item with key 'wrist_img'
if 'wrist_img' in data[0]:
    wrist_img = data[0]['wrist_img']
    print(f"Shape of 'wrist_img': {np.array(wrist_img).shape}")
    # Save the first of the 3x460x612 images into a jpg file
    first_img = wrist_img[-1, 0]  # Select the last image
    first_img = first_img * 255
    print(first_img)
    first_img = np.transpose(first_img, (1, 2, 0))  # Transpose to shape (460, 612, 3)
    cv2.imwrite('cv2_img.jpg', first_img)
    # img = Image.fromarray(first_img.astype('uint8'))  # Convert to PIL Image
    # img.save('eval_wrist_img.jpg')  # Save the image
else:
    print("Key 'wrist_img' not found in the data")


