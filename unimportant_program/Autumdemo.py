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



def check_bend(Motors):
    for i in range(25):
        Motors.move_to_point(4,10*i)
    for i in range(25):
        Motors.move_to_point(4,250 - 10*i)
    for i in range(25):
        Motors.move_to_point(1,10*i)
    for i in range(25):
        Motors.move_to_point(1,250 - 10*i)
    for i in range(25):
        Motors.move_to_point(2,10*i)
    for i in range(25):
        Motors.move_to_point(2,250 - 10*i)    
    
      

    



  
  
Motors = MyDynamixel()

Motors.manual_move()
check_bend(Motors)
