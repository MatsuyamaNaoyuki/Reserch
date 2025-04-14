import threading, csv, os
import time, datetime, getopt, sys
import queue
import pickle

# ----------------------------------------------------------------------------------------

result_dir = r"0408_newfinger_hit"


# ----------------------------------------------------------------------------------------



base_path = r"C:\Users\shigf\Program\data"




basepath = os.path.join(base_path, result_dir)
print(base_path)