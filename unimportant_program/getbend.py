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

def check_bend(Motors, Mc):
    for j in (3, 1, 2):
        kijun = [0, 50, 100, 150, 200, 250]
        i = 0
        p = Motors.get_present_angles()
        while i < 6:
            Motors.move(j,5)
            time.sleep(0.1)
            p = Motors.get_present_angles()
            print(p)
            if p[j-1] > kijun[i]:
                time.sleep(2)
                data , now = Mc.get_data()
                Mc.store_data(data, now)
                i = i+1
                Motors.record_angle()


        p = Motors.get_present_angles()
        while p[j - 1] > 0:
            Motors.move(j,-1)
            p = Motors.get_present_angles()
            print(p)
        Motors.back_to_initial_position()
    
    

Mc = MotionCapture()
Motors = MyDynamixel()
Motors.back_to_initial_position()

Motors.manual_move()
# Motors.back_to_initial_position()

# check_bend(Motors, Mc)
# filename = 'nosensor_motor'
# now = datetime.datetime.now()
# filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.csv'

# with open(filename, 'w',newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(Motors.anglerecord)


# filename = 'nosensor_Mc'
# now = datetime.datetime.now()
# filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.csv'

# with open(filename, 'w',newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(Mc.datas)


