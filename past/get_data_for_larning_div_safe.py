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

# 動作進捗を記録する変数
move_index = 0  # `change_angle` のどこまで進んだかを記録


def init_motion_capture():
    serverIp = '10.1.1.198'
    try:
        opts, args = getopt.getopt([],"hs:",["server="])
    except getopt.GetoptError:
        print('NokovSDKClient.py -s <serverIp>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('NokovSDKClient.py -s <serverIp>')
            sys.exit()
        elif opt in ("-s", "--server"):
            serverIp = arg

    print('serverIp is %s' % serverIp)
    print("Started the Nokov_SDK_Client Demo")
    global client
    client = PySDKClient()

    ver = client.PyNokovVersion()
    
    ret = client.Initialize(bytes(serverIp, encoding="utf8"))
    if ret == 0:
        print("Connect to the Nokov Succeed")
    else:
        print("Connect Failed: [%d]" % ret)
        raise RuntimeError("Failed to connect to motion capture system.")


def safe_thread(func):
    """
    スレッド内で例外をキャッチして、停止イベントをトリガーするデコレータ
    """
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            stop_event.set()  # 停止イベントをトリガー
    return wrapper


@safe_thread
def get_dynamixel(Motors):
    motor_datas = []
    filepath = "C:\\Users\\shigf\\Program\\data\\sentan\\motor_"
    filenumber = 1
    while not stop_event.is_set():  # stop_eventがセットされるまでループ
        now_time = datetime.datetime.now()
        motor_angle = Motors.get_present_angles()
        motor_current = Motors.get_present_currents()
        motor_data = myfunction.combine_lists(motor_angle, motor_current)
        motor_data.insert(0, now_time)
        motor_datas.append(motor_data)
        if write_pkl_event_motor.is_set():
            now = datetime.datetime.now()
            filename = filepath + str(filenumber) + "_" + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
            with open(filename, "wb") as f:
                pickle.dump(motor_datas, f)
            write_pkl_event_motor.clear()
            filenumber += 1
    thread_name = threading.current_thread().name
    results[thread_name] = motor_datas


@safe_thread
def get_magsensor(Ms):
    Mag_datas = []
    filepath = "C:\\Users\\shigf\\Program\\data\\sentan\\mag_"
    filenumber = 1
    while not stop_event.is_set():
        now_time = datetime.datetime.now()
        mag_data = Ms.get_value()
        mag_data = [mag_data]
        mag_data.insert(0, now_time)
        Mag_datas.append(mag_data)
        if write_pkl_event_mag.is_set():
            now = datetime.datetime.now()
            filename = filepath + str(filenumber) + "_" + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
            with open(filename, "wb") as f:
                pickle.dump(Mag_datas, f)
            write_pkl_event_mag.clear()
            filenumber += 1
    thread_name = threading.current_thread().name
    results[thread_name] = Mag_datas


@safe_thread
def move(Motors):
    global move_index  # 現在の進捗を共有する
    with open("C:\\Users\\shigf\\Program\\data\\sentan\\howtomove_50020250121_122232.pickle", mode='br') as fi:
        change_angle = pickle.load(fi)
    len_angle = len(change_angle)
    print(f"Resuming from index {move_index}/{len_angle}")

    for i in range(move_index, len(change_angle)):  # 途中から再開
        angles = change_angle[i]
        print(f"Moving to angles: {angles} ({i}/{len_angle})")
        Motors.move_to_points(angles, times=7)
        move_index = i + 1  # 進捗を更新
        if (i + 1) % 2 == 0:
            write_pkl_event_motor.set()
            write_pkl_event_mag.set()
            write_pkl_event_Mc.set()
            time.sleep(5)
    stop_event.set()
    
def py_data_func(pFrameOfMocapData, pUserData):
    if pFrameOfMocapData == None:  
        print("Not get the data frame.\n")
    else:
        frameData = pFrameOfMocapData.contents
        global preFrmNo, curFrmNo 
        curFrmNo = frameData.iFrame
        if curFrmNo == preFrmNo:
            return
        global client
        preFrmNo = curFrmNo

        length = 128
        szTimeCode = bytes(length)
        client.PyTimecodeStringify(frameData.Timecode, frameData.TimecodeSubframe, szTimeCode, length)
        motiondata = [datetime.datetime.now()]
        for iMarkerSet in range(frameData.nMarkerSets):
            markerset = frameData.MocapData[iMarkerSet]
            for iMarker in range(markerset.nMarkers):
                motiondata.extend([markerset.Markers[iMarker][0],markerset.Markers[iMarker][1], markerset.Markers[iMarker][2]] )
    return motiondata
            

@safe_thread
def get_motioncapture(Ms):
    Motion_datas = []
    filepath = "C:\\Users\\shigf\\Program\\data\\sentan\\mc_"
    filenumber = 1
    while not stop_event.is_set():
        frame = client.PyGetLastFrameOfMocapData()
        if frame:
            try:
                motiondata = py_data_func(frame, client)
                Motion_datas.append(motiondata)
            finally:
                client.PyNokovFreeFrame(frame)
        if write_pkl_event_Mc.is_set():
            now = datetime.datetime.now()
            filename = filepath + str(filenumber) + "_" + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
            with open(filename, "wb") as f:
                pickle.dump(Motion_datas, f)
            write_pkl_event_Mc.clear()
            filenumber += 1
    thread_name = threading.current_thread().name
    results[thread_name] = Motion_datas


while True:
    try:
        init_motion_capture()
        Motors = MyDynamixel()
        Ms = MagneticSensor()
        results = {}
        stop_event = threading.Event()
        write_pkl_event_motor = threading.Event()
        write_pkl_event_mag = threading.Event()
        write_pkl_event_Mc = threading.Event()

        thread1 = threading.Thread(target=get_dynamixel, args=(Motors,), name="motor")
        thread2 = threading.Thread(target=get_magsensor, args=(Ms,), name="magsensor")
        thread3 = threading.Thread(target=move, args=(Motors,), name="move")
        thread4 = threading.Thread(target=get_motioncapture, args=(Ms,), name="motioncapture")

        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()

        thread1.join()
        thread2.join()
        thread3.join()
        thread4.join()
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        print("Restarting...")
