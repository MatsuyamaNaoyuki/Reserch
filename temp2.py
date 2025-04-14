import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel3 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
import queue
import pickle
import pandas as pd
# Motor = MyDynamixel()
hitdata = pd.read_pickle(r"C:\Users\shigf\Program\data\0408_newfinger_hit\mag_8_20250411_032907.pickle")
print(hitdata[0])