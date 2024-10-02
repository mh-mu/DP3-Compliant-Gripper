from openhand_node.hands import Model_T42
import time, math
import numpy as np
from klampt.math import vectorops as vo
from klampt.math import so3, se3
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle, os
from icecream import ic

class T42_controller:
    def __init__(self, finger_offsets, port = '/dev/ttyUSB0', data_collection_mode = False):
        self.T = Model_T42(port=port, s1=1, s2=2, dyn_model='XM', s1_min=0, s2_min=0.5)
        self.data_collection_mode = data_collection_mode
        self.finger_offsets = finger_offsets

    def release(self):
        # print(0.06 + self.finger_offsets[0], 0.06 + self.finger_offsets[1])
        self.T.moveMotor(0, 0.06 + self.finger_offsets[0]) #right finger, viewing from camera 
        self.T.moveMotor(1, 0.06 + self.finger_offsets[1]) #left finger, viewing from camera 

        time.sleep(1)

    def close(self):
        if self.data_collection_mode:
            if np.random.random()>0.5:
                self.T.moveMotor(0, self.finger_offsets[0] - 0.055) # 0.29
                self.T.moveMotor(1, self.finger_offsets[1] - 0.055) # 0.56
            else:
                self.T.moveMotor(1, self.finger_offsets[1] - 0.055) 
                self.T.moveMotor(0, self.finger_offsets[0] - 0.055) 
        else:
            self.T.moveMotor(0, self.finger_offsets[0] - 0.055) # 0.29
            self.T.moveMotor(1, self.finger_offsets[1] - 0.055) # 0.56
        time.sleep(1)

    def move_to_zero_positions(self):
        self.T.moveMotor(0, self.finger_offsets[0]) 
        self.T.moveMotor(1, self.finger_offsets[1])

    def read_motor_positions(self):
        '''
        Returns motor position amnt, between designated min and max values, and encoder positions

        Params:
            None
        Returns:
            amnts, encs : list of motor positions and encoder values
        '''
        return self.T.readHand()
