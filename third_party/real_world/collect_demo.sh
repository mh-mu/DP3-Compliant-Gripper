#!/bin/bash

# Navigate to the directory containing the Python script
# cd /home/mei/workspace/DP3-Compliant-Gripper/third_party/real_world/

# Fix webcam focal point and exposure

# wrist camera
v4l2-ctl -d /dev/video2 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video2 --set-ctrl=focus_absolute=20
v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=1
v4l2-ctl -d /dev/video2 --set-ctrl=exposure_time_absolute=400
v4l2-ctl -d /dev/video2 --set-ctrl=white_balance_automatic=0
v4l2-ctl -d /dev/video2 --set-ctrl=white_balance_temperature=3000
v4l2-ctl -d /dev/video2 --set-ctrl=brightness=128

# # third view camera
# v4l2-ctl -d /dev/video9 --set-ctrl=focus_automatic_continuous=0
# v4l2-ctl -d /dev/video9 --set-ctrl=focus_absolute=8
# v4l2-ctl -d /dev/video9 --set-ctrl=auto_exposure=1
# v4l2-ctl -d /dev/video9 --set-ctrl=exposure_time_absolute=400
# v4l2-ctl -d /dev/video9 --set-ctrl=white_balance_automatic=0
# v4l2-ctl -d /dev/video9 --set-ctrl=white_balance_temperature=3200
# v4l2-ctl -d /dev/video9 --set-ctrl=brightness=128

# # gripper camera
# v4l2-ctl -d /dev/video5 --set-ctrl=focus_automatic_continuous=0
# v4l2-ctl -d /dev/video5 --set-ctrl=focus_absolute=30
# v4l2-ctl -d /dev/video5 --set-ctrl=auto_exposure=1
# v4l2-ctl -d /dev/video5 --set-ctrl=exposure_time_absolute=500
# v4l2-ctl -d /dev/video5 --set-ctrl=white_balance_automatic=0
# v4l2-ctl -d /dev/video5 --set-ctrl=white_balance_temperature=3000
# v4l2-ctl -d /dev/video5 --set-ctrl=brightness=128

# Display webcam image using ffplay
ffplay /dev/video2

# Run the Python script
# python3 collect_demo_data_episode.py --env_name press --finger_type rigid --num_episodes 30

