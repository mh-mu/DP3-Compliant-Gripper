cd 3D-Diffusion-Policy

# Fix webcam focal point and exposure
v4l2-ctl --set-ctrl=focus_automatic_continuous=0
v4l2-ctl --set-ctrl=focus_absolute=10
v4l2-ctl --set-ctrl=auto_exposure=0
v4l2-ctl --set-ctrl=exposure_time_absolute=200

python eval_realworld.py 