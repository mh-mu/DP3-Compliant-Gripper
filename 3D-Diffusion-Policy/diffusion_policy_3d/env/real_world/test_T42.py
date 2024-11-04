import time, math
import numpy as np
from klampt.math import vectorops as vo
from klampt.math import so3, se3
from icecream import ic
import sys, os

from T42_controller import T42_controller
from CONSTANTS import finger_zero_positions, gripper_port

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..', 'third_party', 'openhand_node', 'src', 'openhand_node')))
from hands import Model_T42

finger_zero_positions = [0.1, 0.17]

gripper = T42_controller(finger_zero_positions, port=gripper_port, data_collection_mode=False)
# ic(gripper.read_motor_positions())
gripper.release()
time.sleep(1)
# # ic(gripper.read_motor_positions())
# # gripper.move_to_zero_positions()
gripper.close()

# T = Model_T42(port='/dev/ttyUSB0', s1=1, s2=2, dyn_model='XM', s1_min=0.02, s2_min=0.4)
# print(T.readMotor(0))
# print(T.readMotor(1))
# time.sleep(1)
# # T.release()
# # # T.diagnostics()
# T.moveMotor(0, 0.2)
# T.moveMotor(1, 0.2)
# # T.motorDir = [0, 0]
# # T.moveMotor(0, 0.2)
# # T.release()
# ic(T.readHand())