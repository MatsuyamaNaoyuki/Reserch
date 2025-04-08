import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel3 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
import queue
import pickle
import pandas as pd
# Motor = MyDynamixel()
hitdata = pd.read_pickle(r"C:\Users\shigf\Program\Reserch\mixhit_fortest20250227_135056.pickle")
print(len(hitdata))
nohitdata = pd.read_pickle(r"C:\Users\shigf\Program\data\withhit\testyou\nohit_fortest20250227_134852.pickle")
print(len(nohitdata))
df_vertical = pd.concat([hitdata, nohitdata], axis=0)
print(len(df_vertical))
myfunction.wirte_pkl(df_vertical, "mixhit_fortest")
