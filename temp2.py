import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction

import pickle




result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\alluse_stride20"

base_dir = os.path.dirname(result_dir)
kijun_dir = myfunction.find_pickle_files("kijun", base_dir)

print(kijun_dir)
