cd 3D-Diffusion-Policy

# Fix webcam focal point and exposure
# v4l2-ctl -d /dev/video2 --set-ctrl=focus_automatic_continuous=0
# v4l2-ctl -d /dev/video2 --set-ctrl=focus_absolute=20
# v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=1
# v4l2-ctl -d /dev/video2 --set-ctrl=exposure_time_absolute=300

v4l2-ctl -d /dev/video3 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video3 --set-ctrl=focus_absolute=20
v4l2-ctl -d /dev/video3 --set-ctrl=auto_exposure=1
v4l2-ctl -d /dev/video3 --set-ctrl=exposure_time_absolute=100

# Display webcam image using ffplay
# ffplay /dev/video3

python eval_realworld.py 