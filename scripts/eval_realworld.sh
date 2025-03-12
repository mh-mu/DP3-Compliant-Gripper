cd 3D-Diffusion-Policy

# # wrist camera
# v4l2-ctl -d /dev/video2 --set-ctrl=focus_automatic_continuous=0
# v4l2-ctl -d /dev/video2 --set-ctrl=focus_absolute=20
# v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=1
# v4l2-ctl -d /dev/video2 --set-ctrl=exposure_time_absolute=80
# v4l2-ctl -d /dev/video2 --set-ctrl=white_balance_automatic=0
# v4l2-ctl -d /dev/video2 --set-ctrl=white_balance_temperature=3000
# v4l2-ctl -d /dev/video2 --set-ctrl=brightness=128

# # gripper camera
# v4l2-ctl -d /dev/video8 --set-ctrl=focus_automatic_continuous=0
# v4l2-ctl -d /dev/video8 --set-ctrl=focus_absolute=30
# v4l2-ctl -d /dev/video8 --set-ctrl=auto_exposure=1
# v4l2-ctl -d /dev/video8 --set-ctrl=exposure_time_absolute=70
# v4l2-ctl -d /dev/video8 --set-ctrl=white_balance_automatic=0
# v4l2-ctl -d /dev/video8 --set-ctrl=white_balance_temperature=3200
# v4l2-ctl -d /dev/video8 --set-ctrl=brightness=128

# # third_view camera
# v4l2-ctl -d /dev/video5 --set-ctrl=focus_automatic_continuous=0
# v4l2-ctl -d /dev/video5 --set-ctrl=focus_absolute=10
# v4l2-ctl -d /dev/video5 --set-ctrl=auto_exposure=1
# v4l2-ctl -d /dev/video5 --set-ctrl=exposure_time_absolute=70
# v4l2-ctl -d /dev/video5 --set-ctrl=white_balance_automatic=0
# v4l2-ctl -d /dev/video5 --set-ctrl=white_balance_temperature=3000
# v4l2-ctl -d /dev/video5 --set-ctrl=brightness=128

# Display webcam image using ffplay
# ffplay /dev/video2

export HYDRA_FULL_ERROR=1

python eval_realworld.py 