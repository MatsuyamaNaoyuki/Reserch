import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel3 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
import queue
import pickle
import pandas as pd
Motor = MyDynamixel()
data = pd.read_pickle(r"C:\Users\shigf\Program\data\hit_test\howtomove_50020250127_164123.pickle")

data = data[0:10]
myfunction.wirte_pkl(data, "howtomove")