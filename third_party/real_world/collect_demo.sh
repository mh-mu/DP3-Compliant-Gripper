#!/bin/bash

# Navigate to the directory containing the Python script
cd /home/mei/workspace/DP3-Compliant-Gripper/third_party/real_world/

# Fix webcam focal point and exposure
v4l2-ctl -d /dev/video2 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video2 --set-ctrl=focus_absolute=20
v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=1
v4l2-ctl -d /dev/video2 --set-ctrl=exposure_time_absolute=300

# v4l2-ctl -d /dev/video3 --set-ctrl=focus_automatic_continuous=0
# v4l2-ctl -d /dev/video3 --set-ctrl=focus_absolute=20
# v4l2-ctl -d /dev/video3 --set-ctrl=auto_exposure=1
# v4l2-ctl -d /dev/video3 --set-ctrl=exposure_time_absolute=100

# Display webcam image using ffplay
# ffplay /dev/video2

# Run the Python script
python3 collect_demo_data_episode.py --env_name contact_compliant --finger_type compliant --num_episodes 30

