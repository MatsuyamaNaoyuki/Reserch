import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel3 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
import queue
import pickle
import pandas as pd
data1 = pd.read_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger_mixhit\tube_softfinger_hit_1500_20250411_143303.pickle")

data2 = pd.read_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger_mixhit\tube_softfinger_nohit_1500_20250411_144433.pickle")
