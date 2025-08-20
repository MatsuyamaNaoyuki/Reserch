import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel4 import MyDynamixel
import pickle





        
def get_dynamixel(Motors, motorpath):
    motor_datas = []
    filepath = motorpath
    filenumber = 1
    i = 0
    try:
        while not stop_event.is_set():  # stop_eventがセットされるまでループ
            now_time  = datetime.datetime.now()
            motor_angle = Motors.get_present_angles()
            motor_current = Motors.get_present_currents()
            # formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            motor_data = myfunction.combine_lists(motor_angle, motor_current)
            motor_data.insert(0, now_time)
            motor_datas.append(motor_data)
            if write_pkl_event_motor.is_set():
                print(motor_data)
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


def move(Motors, howtomovepath):
    with open(howtomovepath, mode='br') as fi:
        change_angle = pickle.load(fi)
    len_angle = len(change_angle)
    print(change_angle)
    time_len = None
    # time_len = [4,4,10,10,7,7, 5,5, 9,9]
    for i, angles in enumerate(change_angle):
        print(angles)
        print(str(i) + "/" +  str(len_angle))
        if time_len is None:
            Motors.move_to_points(angles, times = 7)
        else:
            Motors.move_to_points(angles, times = time_len[i%len(time_len)])
        if i% 1000 == 0:
            write_pkl_event_motor.set()
            write_pkl_event_mag.set()
            write_pkl_event_Mc.set()
            time.sleep(5)

    # time.sleep(2)
    stop_event.set()
    
    





# ----------------------------------------------------------------------------------------


# result_dir = r"0520\nohit1500kai"

base_path = r"C:\Users\shigf\Program\data\temp"

# ----------------------------------------------------------------------------------------








print(base_path)
motorpath = os.path.join(base_path, "motor_")
magpath = os.path.join(base_path, "mag_")
mcpath = os.path.join(base_path, "mc_")
howtomovepath = myfunction.find_pickle_files("howtomove", base_path)


Motors = MyDynamixel()
results = {}
stop_event = threading.Event()
write_pkl_event_motor = threading.Event()
write_pkl_event_mag = threading.Event()
write_pkl_event_Mc = threading.Event()

print("◆スレッド:",threading.current_thread().name)

thread1 = threading.Thread(target=get_dynamixel, args=(Motors,motorpath,), name="motor")
thread3 = threading.Thread(target=move, args=(Motors,howtomovepath,), name="move")


thread1.start()
thread3.start()


thread1.join()
thread3.join()


print("A")
for key, value in results.items():
    print("IN")
    filename = key
    now = datetime.datetime.now()
    filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
    with open(filename, 'wb') as fo:
        pickle.dump(value, fo)
    





# print("Results:", results)