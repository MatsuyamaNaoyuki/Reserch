#マニュアルムーブするだけ

import sys,os, time, datetime, getopt
sys.path.append(os.path.join(os.path.dirname(__file__), 'myclass'))
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting
import csv
import pprint
from myclass.MyDynamixel4 import MyDynamixel
# from myclass.MotionCapture2 import MotionCapture
from myclass import myfunction
import pickle
from nokov.nokovsdk import *

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

def get_motiondata():          
    frame = client.PyGetLastFrameOfMocapData()
    if frame :
        try:
            motiondata = py_data_func(frame, client)
        finally:
            client.PyNokovFreeFrame(frame)
    return motiondata


Motors = MyDynamixel()
init_motion_capture()

Motors.move_to_points([-10, -10, -10])


dif = 0
threshold_value = 10

base_motion_data = get_motiondata()
for i in range(len(list(Motors.IDs))):
    while dif > threshold_value:
        Motors.move(i, 1)
        now_motiondata = get_motiondata()
        dif = now_motiondata - base_motion_data

    Motors.move(i, -1)
    base_motion_data = get_motiondata()


