#一部分のフィンガーに対して値を取得する用
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
from myclass.MyMagneticSensor import MagneticSensor


def combine_lists(*lists):
    combined_list = []
    for lst in lists:
        combined_list.extend(lst)
    return combined_list

def check_bend(Motors, Mag):
    datas = []
    for i in range(150):
        Motors.move_to_point(4,i)
        time.sleep(0.1)
        now_time  = datetime.datetime.now()
        motor_angle = Motors.get_present_angles()
        mag_data = Mag.get_value()
        formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        mag_data = Mag.change_data(mag_data)
        all_data = combine_lists(motor_angle, mag_data)
        all_data.insert(0, formatted_now)
        datas.append(all_data)
        print(i)
    for i in range(150):
        Motors.move_to_point(4, (150 -i) )
        time.sleep(0.1)
        now_time  = datetime.datetime.now()
        motor_angle = Motors.get_present_angles()
        mag_data = Mag.get_value()
        formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        mag_data = Mag.change_data(mag_data)
        all_data = combine_lists(motor_angle, mag_data)
        all_data.insert(0, formatted_now)
        datas.append(all_data)
        print(i)
    for i in range(150):
        Motors.move_to_point(1,i)
        Motors.move_to_point(2,-0.4 * i)
        time.sleep(0.1)
        now_time  = datetime.datetime.now()
        motor_angle = Motors.get_present_angles()
        mag_data = Mag.get_value()
        formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        mag_data = Mag.change_data(mag_data)
        all_data = combine_lists(motor_angle, mag_data)
        all_data.insert(0, formatted_now)
        datas.append(all_data)
        print(i)
    for i in range(150):
        Motors.move_to_point(1, (150 -i) )
        Motors.move_to_point(2, (150 -i) * -0.4)
        time.sleep(0.1)
        now_time  = datetime.datetime.now()
        motor_angle = Motors.get_present_angles()
        mag_data = Mag.get_value()
        formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        mag_data = Mag.change_data(mag_data)
        all_data = combine_lists(motor_angle, mag_data)
        all_data.insert(0, formatted_now)
        datas.append(all_data)
        print(i)
    for i in range(150):
        Motors.move_to_point(2,i)
        Motors.move_to_point(1,-0.4 * i )
        time.sleep(0.1)
        now_time  = datetime.datetime.now()
        motor_angle = Motors.get_present_angles()
        mag_data = Mag.get_value()
        formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        mag_data = Mag.change_data(mag_data)
        all_data = combine_lists(motor_angle, mag_data)
        all_data.insert(0, formatted_now)
        datas.append(all_data)
        print(i)
    for i in range(150):
        Motors.move_to_point(2, (150 -i) )
        Motors.move_to_point(1, (150 -i) * -0.4 )
        time.sleep(0.1)
        now_time  = datetime.datetime.now()
        motor_angle = Motors.get_present_angles()
        mag_data = Mag.get_value()
        formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        mag_data = Mag.change_data(mag_data)
        all_data = combine_lists(motor_angle, mag_data)
        all_data.insert(0, formatted_now)
        datas.append(all_data)
        print(i)
    return datas


  

Motors = MyDynamixel()
Mag = MagneticSensor()
# Motors.back_to_initial_position()
Motors.manual_move()
Motors.set_start_angle()
data = check_bend(Motors, Mag)
# data = myfunction.get_all_data(Motors, Mc, Mag)
print(data)

filename = 'Autaum_mid_data_harubaha_farsand'
now = datetime.datetime.now()
filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.csv'

with open(filename, 'w',newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)


  


    
