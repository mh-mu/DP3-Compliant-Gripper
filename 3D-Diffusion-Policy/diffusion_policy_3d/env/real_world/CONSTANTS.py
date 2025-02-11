import math
# t_pickup_high = [-0.485, -0.178, 0.44] 
# t_pickup = [-0.485, -0.178, 0.35] 
# # t_wipe_start = [0.4, -0.02, 0.28]
# t_random_start = [-0.57, 0.0075, 0.34]
# t_fixed_start = [-0.57, 0.0075, 0.34]

t_pickup_high = [-0.485, -0.19, 0.22] 
t_pickup = [-0.485, -0.19, 0.13] 
# t_wipe_start = [0.4, -0.02, 0.28]
t_random_start = [-0.585, 0.00, 0.155] # this is for 35 mm #[-0.585, 0.00, 0.1615]
t_fixed_start = [-0.56, 0.0075, 0.13]
R_default = [0, 0, -1, math.sqrt(2)/2, math.sqrt(2)/2, 0, math.sqrt(2)/2, -math.sqrt(2)/2, 0]

R = [-0.036093246883230824, -0.0016417553230183065, -0.9993470779308286, 0.8343056200142569, 0.5504278824056338, -0.03103673119186873, 0.5501194506123673, -0.8348810998638722, -0.018497003758397978]
t = [-0.5719669638367115, -0.13510426694812794, 0.00818264088493828]

ft_ip = 'http://192.168.0.102:80'
gripper_port = '/dev/ttyUSB1'
# note: larger -> lose, small -> close
finger_offset_positions_compliant = [0.15, 0.15]# [0.2, 0.14]# [0.1, 0.17] #[0.0675, 0.17] #[0.0725, 0.175] #[0.075, 0.18] #[0.065, 0.175] #[0.0775, 0.1845] #[0.08, 0.1875] #[0.085, 0.19] (2nd, current calibration pic) #[0.095, 0.2] (initial)
finger_offset_positions_rigid = [0.11, 0.11]

UR5_ip = '192.168.0.101'
UR5_home_position = (R, t)
R_EE_WORLD_HOME = [0,0,-1, math.sqrt(2)/2,math.sqrt(2)/2,0,math.sqrt(2)/2,-math.sqrt(2)/2,0] #klampt format
R_ATI_EE = [0, math.sqrt(2)/2,math.sqrt(2)/2, 0,math.sqrt(2)/2,-math.sqrt(2)/2, 1, 0, 0]
HOME_t_obj = [-0.6, 0, 0.08] #[-0.5, 0, 0.08]

# gripper actions
OPEN = 0
CLOSE = 1