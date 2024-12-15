#MagSensorを用いた学習のデータセット作成のプログラム（予定）
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
from myclass.MyMagneticSensor import MagneticSensor



def check_bend(Motors, Mc, Mag):
    datas = []
    for i in range(150):
        Motors.move_to_point(4,2*i)
        data = myfunction.get_all_data(Motors, Mc, Mag)
        datas.append(data)
        print(i)
    for i in range(150):
        Motors.move_to_point(4,(150 - i)*2)
        data = myfunction.get_all_data(Motors, Mc, Mag)
        datas.append(data)
        print(i)
    for i in range(150):
        Motors.move_to_point(1,i*2)

        data = myfunction.get_all_data(Motors, Mc, Mag)
        datas.append(data)
        print(i)
    for i in range(150):
        Motors.move_to_point(1,2*(150 - i))
        # time.sleep(0.1)
        data = myfunction.get_all_data(Motors, Mc, Mag)
        datas.append(data)
        print(i)
    for i in range(150):
        Motors.move_to_point(2,2* i)
        # time.sleep(0.1)
        data = myfunction.get_all_data(Motors, Mc, Mag)
        datas.append(data)
        print(i)
    for i in range(150):
        Motors.move_to_point(2,(150 - i)*2)
        # time.sleep(0.1)
        data = myfunction.get_all_data(Motors, Mc, Mag)
        datas.append(data)
        print(i)
    return datas


  
  
Mc = MotionCapture()
Motors = MyDynamixel()
Mag = MagneticSensor()
Motors.manual_move()
data = myfunction.get_all_data(Motors, Mc, Mag)


print(data)

filename = 'pictureyou'
now = datetime.datetime.now()
filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.csv'

with open(filename, 'w',newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)