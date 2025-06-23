import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel2 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
import queue
import pickle
import keyboard
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

def move(Motors):
    Motors.manual_move()
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
            


def get_motioncapture(mcpath):
    Motion_datas = []
    try:
        while not stop_event.is_set():
            if keyboard.is_pressed('z'):
                frame = client.PyGetLastFrameOfMocapData()
                if frame:
                    try:
                        motiondata = py_data_func(frame, client)
                        Motion_datas.append(motiondata)
                    finally:
                        client.PyNokovFreeFrame(frame)
                    
                    # 連続取得を防ぐために zキーが離されるのを待つ
                    while keyboard.is_pressed('z') and not stop_event.is_set():
                        pass

    finally:
        thread_name = threading.current_thread().name
        results[thread_name] = Motion_datas
        



# ----------------------------------------------------------------------------------------


# result_dir = r"0520\nohit1500kai"

base_path = r"C:\Users\shigf\Program\data\0520\hitfortest"

# ----------------------------------------------------------------------------------------








print(base_path)

mcpath = os.path.join(base_path, "mc_")


init_motion_capture()
Motors = MyDynamixel()

results = {}
stop_event = threading.Event()
write_pkl_event_motor = threading.Event()
write_pkl_event_mag = threading.Event()
write_pkl_event_Mc = threading.Event()

print("◆スレッド:",threading.current_thread().name)


thread3 = threading.Thread(target=move, args=(Motors,), name="move")
thread4 = threading.Thread(target=get_motioncapture, args=(mcpath,), name="motioncapture")


thread3.start()
thread4.start()


thread3.join()
thread4.join()


for key, value in results.items():
    filename = key
    now = datetime.datetime.now()
    filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
    with open(filename, 'wb') as fo:
        pickle.dump(value, fo)
    





print("Results:", results)