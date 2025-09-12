#マニュアルムーブするだけ

import time, datetime, getopt, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'myclass'))
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting
import csv
import pprint, math
from myclass.MyDynamixel4 import MyDynamixel
# from myclass.MotionCapture2 import MotionCapture
from myclass import myfunction
from nokov.nokovsdk import *
preFrmNo = 0
curFrmNo = 0
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

def get_motioncapture():        
    frame = client.PyGetLastFrameOfMocapData()
    if frame :
        try:
            motiondata = py_data_func(frame, client)
        finally:
            client.PyNokovFreeFrame(frame)

    return motiondata[-3:]

def culc_dif(basemotion, nowmotion):
    difsum = 0
    for i in range(len(basemotion)):
        dif = (basemotion[i] - nowmotion[i])**2
        difsum = difsum + dif
    return math.sqrt(difsum)
init_motion_capture()
Motors = MyDynamixel()


Motors.move_to_points([-10,-10,-10], times=10)
base_motion = get_motioncapture()

base_motion = get_motioncapture()

dif = 0
dif_sikii = 3


for i in range(len(list(Motors.IDs))):
    print(i)
    while dif < dif_sikii:
        Motors.move(i + 1, 1)
        time.sleep(0.1)
        now_motion = get_motioncapture()
        dif = culc_dif(base_motion, now_motion)
    Motors.move(i+1, -1)
    dif = 0
    base_motion = get_motioncapture()