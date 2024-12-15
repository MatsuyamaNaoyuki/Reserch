#とってきた点群を三次元に表示するやつ（たぶん使わない）


import sys,os, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'myclass'))
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting
import csv
import pprint
from myclass.MyDynamixel2 import MyDynamixel
from myclass.MotionCapture2 import MotionCapture
from myclass import myfunction
import pandas as pd
import numpy as np
import random

def swap(a, b):
    return b,a

path = r'C:\Users\shigf\Program\DXhub\sensor_Mc20240719_113028.csv'

sensorcoordinate = myfunction.read_coordinate(path)
sensoravereage = myfunction.get_avereage(sensorcoordinate)
sensornew = sensorcoordinate - sensoravereage
np.set_printoptions(suppress=True)

path = r'C:\Users\shigf\Program\DXhub\nosensor_Mc20240719_104116.csv'

nosensorcoordinate = myfunction.read_coordinate(path)
nosensoravereage = myfunction.get_avereage(nosensorcoordinate)
nosensornew = nosensorcoordinate - nosensoravereage
print(sensornew[5][0])
print(sensornew[5][3])
sensornew[5][0], sensornew[5][5] = sensornew[5][5].copy(), sensornew[5][0].copy()
sensornew[5][4], sensornew[5][1] = sensornew[5][1].copy(), sensornew[5][4].copy()
sensornew[5][3], sensornew[5][2] = sensornew[5][2].copy(), sensornew[5][3].copy()
sensornew[5][4], sensornew[5][5] = sensornew[5][5].copy(), sensornew[5][4].copy()
nosensornew[5][0], nosensornew[5][5] = nosensornew[5][5].copy(), nosensornew[5][0].copy()
nosensornew[5][4], nosensornew[5][1] = nosensornew[5][1].copy(), nosensornew[5][4].copy()
# nosensornew[5][3], nosensornew[5][2] = nosensornew[5][2].copy(), nosensornew[5][3].copy()
nosensornew[5][4], nosensornew[5][5] = nosensornew[5][5].copy(), nosensornew[5][4].copy()
print(sensornew)
print(sensornew[5][3])
coor = np.array([sensornew[5], nosensornew[5]])
np.set_printoptions(suppress=True)
myfunction.make_3D_graphs(coor, labelname=["sensor", "nosensor"], lineswitch= True)