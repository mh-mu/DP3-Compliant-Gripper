#!/bin/bash

# Navigate to the directory containing the Python script
cd /home/mei/workspace/DP3-Compliant-Gripper/third_party/real_world/

# Fix webcam focal point and exposure
v4l2-ctl --set-ctrl=focus_automatic_continuous=0
v4l2-ctl --set-ctrl=focus_absolute=10
v4l2-ctl --set-ctrl=auto_exposure=0
v4l2-ctl --set-ctrl=exposure_time_absolute=200

# Display webcam image using ffplay
# ffplay /dev/video0

# Run the Python script
python3 collect_demo_data_episode.py --env_name easy --finger_type rigid --num_episodes 30

