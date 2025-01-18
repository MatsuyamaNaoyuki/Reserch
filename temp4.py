import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel2 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
import queue
import pickle
from nokov.nokovsdk import *
preFrmNo = 0
curFrmNo = 0
global client



def get_dynamixel():
    motor_datas = []
    filepath = "C:\\Users\\shigf\\Program\\data\\motor_"
    filenumber = 1
    i = 0
    try:
        while not stop_event.is_set():  # stop_eventがセットされるまでループ
            i = i + 1
            motor_datas.append(i)
            if write_pkl_event_motor.is_set():
                # writing_motor_event.set()
                now = datetime.datetime.now()
                filename = filepath + str(filenumber) + "_" +now.strftime('%Y%m%d_%H%M%S') + '.pickle'
                with open(filename, "wb") as f:
                    pickle.dump(motor_datas, f)  # motor_datasをpickleで保存
                write_pkl_event_motor.clear()  # write_pkl_eventをリセット
                filenumber = filenumber + 1
                # writing_motor_event.clear()
    finally:
        thread_name = threading.current_thread().name
        results[thread_name] = motor_datas

# daemon=Trueで強制終了

def get_magsensor():
    Mag_datas = []
    filepath = "C:\\Users\\shigf\\Program\\data\\mag_"
    filenumber = 1
    i = 0
    try:
        while not stop_event.is_set():  # stop_eventがセットされるまでループ\
            i = i + 1
            Mag_datas.append(i)
            if write_pkl_event_mag.is_set():
                # writing_mag_event.set()
                now = datetime.datetime.now()
                filename = filepath + str(filenumber) + "_"+ now.strftime('%Y%m%d_%H%M%S') + '.pickle'
                with open(filename, "wb") as f:
                    pickle.dump(Mag_datas, f)  # motor_datasをpickleで保存
                write_pkl_event_mag.clear()  # write_pkl_eventをリセット
                filenumber = filenumber + 1
                # writing_mag_event.clear()
    finally:
        thread_name = threading.current_thread().name
        results[thread_name] = Mag_datas

    

def move():

    change_angle = range(10)
    for i, len in enumerate(change_angle):

        print(len)


        if (i +1) % 3 == 0:
            write_pkl_event_motor.set()
            write_pkl_event_mag.set()
            write_pkl_event_Mc.set()
            time.sleep(5)

    # time.sleep(2)
    stop_event.set()
    
    
    

        





# init_motion_capture()
# Motors = MyDynamixel()
# Ms = MagneticSensor()
results = {}
stop_event = threading.Event()
write_pkl_event_motor = threading.Event()
write_pkl_event_mag = threading.Event()
write_pkl_event_Mc = threading.Event()
# writing_motor_event = threading.Event()
# writing_mag_event = threading.Event()
# writing_Mc_event = threading.Event()

print("◆スレッド:",threading.current_thread().name)

thread1 = threading.Thread(target=get_dynamixel, name="motor")
thread2 = threading.Thread(target=get_magsensor, name="magsensor")
thread3 = threading.Thread(target=move, name="move")
# thread4 = threading.Thread(target=get_motioncapture, args=(Ms,), name="motioncapture")

thread1.start()
thread2.start()
thread3.start()
# thread4.start()

thread1.join()
thread2.join()
thread3.join()
# thread4.join()


# for key, value in results.items():
#     filename = key
#     now = datetime.datetime.now()
#     filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
#     with open(filename, 'wb') as fo:
#         pickle.dump(value, fo)
    



# for key, value in results.items():
#     filename = key
#     now = datetime.datetime.now()
#     filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.csv'
#     with open(filename, 'w',newline="") as f:
#         writer = csv.writer(f)
#         writer.writerows(value)
    

# print("Results:", results)