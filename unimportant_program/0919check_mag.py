#磁石の値を動かすだけ，動作は外力


import sys,os, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'myclass'))
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting
import csv
import pprint
from myclass import myfunction
from myclass.MyMagneticSensor import MagneticSensor


                

    

Mags = MagneticSensor()
key = ''
kb = kbhit.KBHit()
i = 0
while key != 'q':   # 'q'が押されると終了 
    if kb.kbhit():
        key = kb.getch()
    sensor_value = Mags.get_value()
    now_time  = datetime.datetime.now()
    i = i + 1
    if i % 10 == 0:
        print(sensor_value)
    floatdata = Mags.change_data(sensor_value)
    Mags.store_data(floatdata, now_time)





filename = 'MagChack'
now = datetime.datetime.now()
filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.csv'

with open(filename, 'w',newline="") as f:
    writer = csv.writer(f)
    writer.writerows(Mags.datas)



