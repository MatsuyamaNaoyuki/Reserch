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

    print ('serverIp is %s' % serverIp)
    print("Started the Nokov_SDK_Client Demo")
    global client
    client = PySDKClient()

    ver = client.PyNokovVersion()
    
    ret = client.Initialize(bytes(serverIp, encoding = "utf8"))
    if ret == 0:
        print("Connect to the Nokov Succeed")
    else:
        print("Connect Failed: [%d]" % ret)
        exit(0)
        
def get_dynamixel(Motors):
    motor_datas = []
    try:
        while not stop_event.is_set():  # stop_eventがセットされるまでループ
            now_time  = datetime.datetime.now()
            motor_angle = Motors.get_present_angles()
            motor_current = Motors.get_present_currents()
            # formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            motor_data = myfunction.combine_lists(motor_angle, motor_current)
            motor_data.insert(0, now_time)
            motor_datas.append(motor_data)
    finally:
        thread_name = threading.current_thread().name
        results[thread_name] = motor_datas
        print("dy")

# daemon=Trueで強制終了

def get_magsensor(Ms):
    Mag_datas = []
    try:
        while not stop_event.is_set():  # stop_eventがセットされるまでループ\
            now_time  = datetime.datetime.now()
            mag_data = Ms.get_value()
            # formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            mag_data = [mag_data]
            mag_data.insert(0, now_time)
            Mag_datas.append(mag_data)
    finally:
        thread_name = threading.current_thread().name
        results[thread_name] = Mag_datas
        print("mag")

    

def move(Motors):
    with open("C:\\Users\\shigf\\Program\\data\\howtomove_0109_2d20250116_190231.pickle", mode='br') as fi:
        change_angle = pickle.load(fi)
    
    for len in change_angle:
        print(len)
        Motors.move_to_point(3, len[2])
        time.sleep(0.2)

    # time.sleep(2)
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
            


def get_motioncapture(Ms):
    Motion_datas = []
    i = 0
    try:
        while not stop_event.is_set():  # stop_eventがセットされるまでループ\
            frame = client.PyGetLastFrameOfMocapData()
            if frame :
                try:
                    motiondata = py_data_func(frame, client)
                    print(motiondata)
                    Motion_datas.append(motiondata)
                finally:
                    client.PyNokovFreeFrame(frame)
    finally:
        thread_name = threading.current_thread().name
        results[thread_name] = Motion_datas
        print("motion")
        
        





init_motion_capture()
Motors = MyDynamixel()
Ms = MagneticSensor()
results = {}
stop_event = threading.Event()

print("◆スレッド:",threading.current_thread().name)

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


for key, value in results.items():
    filename = key
    now = datetime.datetime.now()
    filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
    with open(filename, 'wb') as fo:
        pickle.dump(value, fo)
    



for key, value in results.items():
    filename = key
    now = datetime.datetime.now()
    filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.csv'
    with open(filename, 'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerows(value)
    

print("Results:", results)