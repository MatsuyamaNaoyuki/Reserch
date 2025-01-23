#マニュアルムーブするだけ

import sys,os, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'myclass'))
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting
import csv
import pprint
from myclass.MyDynamixel2 import MyDynamixel
# from myclass.MotionCapture2 import MotionCapture
from myclass import myfunction

Motors = MyDynamixel()
Motors.manual_move()


Motors.back_to_initial_position()




# filename = 'nosensor_Mc'q
# now = datetime.datetime.now()
# filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.csv'

# with open(filename, 'w',newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(Mc.datas)ttqqw


