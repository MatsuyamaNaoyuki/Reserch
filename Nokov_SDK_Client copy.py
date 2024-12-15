__author__ = 'duguguang'

from nokov.nokovsdk import *
import time
import sys, getopt, datetime
from threading import Thread

preFrmNo = 0
curFrmNo = 0
global client

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
        # print(f"\nFrameNo: {frameData.iFrame}\tTimeStamp:{frameData.iTimeStamp}\t Timecode:{szTimeCode.decode('utf-8')}")					
        # print( f"MarkerSet [Count={frameData.nMarkerSets}]")
        motiondata = [datetime.datetime.now()]
        for iMarkerSet in range(frameData.nMarkerSets):
            markerset = frameData.MocapData[iMarkerSet]
            # print( f"Markerset{iMarkerSet+1}: {markerset.szName.decode('utf-8')} [nMarkers Count={markerset.nMarkers}]")
            # print("{")
            for iMarker in range(markerset.nMarkers):
                # print(f"\tMarker{iMarker+1}(mm) \tx:{markerset.Markers[iMarker][0]:6.2f}"\
                #     f"\ty:{markerset.Markers[iMarker][1]:6.2f}\tz:{markerset.Markers[iMarker][2]:6.2f}")
                motiondata.extend([markerset.Markers[iMarker][0],markerset.Markers[iMarker][1], markerset.Markers[iMarker][2]] )
            # print( "}")
        print(motiondata)
            

 
def py_msg_func(iLogLevel, szLogMessage):
    szLevel = "None"
    if iLogLevel == 4:
        szLevel = "Debug"
    elif iLogLevel == 3:
        szLevel = "Info"
    elif iLogLevel == 2:
        szLevel = "Warning"
    elif iLogLevel == 1:
        szLevel = "Error"
  
    print("[%s] %s" % (szLevel, cast(szLogMessage, c_char_p).value))

def py_forcePlate_func(pFocePlates, pUserData):
    if pFocePlates == None:  
        print("Not get the forcePlate frame.\n")
        pass
    else:
        ForcePlatesData = pFocePlates.contents
        print("iFrame:%d" % ForcePlatesData.iFrame)
        for iForcePlate in range(ForcePlatesData.nForcePlates):
            print("Fxyz:[%f,%f,%f] xyz:[%f,%f,%f] MFree:[%f]" % (
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[0],
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[1],
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[2],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[0],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[1],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[2],
                ForcePlatesData.ForcePlates[iForcePlate].Mfree
            ))

def py_analog_channel_func(pAnalogData, pUserData):
    if pAnalogData == None:  
        print("Not get the analog data frame.\n")
        pass
    else:
        anData = pAnalogData.contents
        print(f"\nFrameNO:{anData.iFrame}\tTimeStamp:{anData.iTimeStamp}")
        print(f"Analog Channel Number:{anData.nAnalogdatas}, SubFrame: {anData.nSubFrame}")
        for ch in range(anData.nAnalogdatas):
            print(f"Channel {ch} ", end="")
            for sub in range(anData.nSubFrame):
                print(f",{anData.Analogdata[ch][sub]:6.4f}", end="")
            print("")

def read_data_func(client):
    while True:
        frame = client.PyGetLastFrameOfMocapData()
        if frame :
            try:
                py_data_func(frame, client)
            finally:
                client.PyNokovFreeFrame(frame)

def py_desc_func(pdds):
    dataDefs = pdds.contents

    for iDef in range(dataDefs.nDataDescriptions):
        dataDef = dataDefs.arrDataDescriptions[iDef]
        
        if dataDef.type == DataDescriptors.Descriptor_Skeleton.value:
            skeletonDef = dataDef.Data.SkeletonDescription.contents
            print(f"Skeleton Name:{skeletonDef.szName.decode('utf-8')}, id:{skeletonDef.skeletonID}, rigids:{skeletonDef.nRigidBodies}")
            for iBody in range(skeletonDef.nRigidBodies):
                bodyDef = skeletonDef.RigidBodies[iBody]
                print(f"[{bodyDef.ID}] {bodyDef.szName.decode('utf-8')} {bodyDef.parentID} "\
                    f"{bodyDef.offsetx:.6f}mm {bodyDef.offsety:.6f}mm {bodyDef.offsetz:.6f}mm "\
                    f"{bodyDef.qx:.6f} {bodyDef.qy:.6f} {bodyDef.qz:.6f} {bodyDef.qw:.6f}")
        elif dataDef.type == DataDescriptors.Descriptor_MarkerSet.value:
            markerSetDef = dataDef.Data.MarkerSetDescription.contents
            print(f"MarkerSetName: {markerSetDef.szName.decode('utf-8')}")
            for markerIndex in range(markerSetDef.nMarkers):
                markerName = markerSetDef.szMarkerNames[markerIndex]
                print(f"Marker[{markerIndex}] : {markerName.decode('utf-8')}")
        elif dataDef.type == DataDescriptors.Descriptor_RigidBody.value:
            rigidBody = dataDef.Data.RigidBodyDescription.contents
            print(f"RigidBody:{rigidBody.szName.decode('utf-8')} ID:{rigidBody.ID}")
        elif dataDef.type == DataDescriptors.Descriptor_ForcePlate.value:
            forcePlateDef = dataDef.Data.ForcePlateDescription.contents
            for chIndex in range(forcePlateDef.nChannels):
                channelName = forcePlateDef.szChannelNames[chIndex].value.decode('utf-8')
                print(f"Channel:{chIndex} {channelName}")
        elif dataDef.type == DataDescriptors.Descriptor_Param.value:
            dataParam = dataDef.Data.DataParam.contents
            print(f'FrameRate:{dataParam.nFrameRate}')

def py_notify_func(pNotify, userData):
    notify = pNotify.contents
    print(f"\nNotify Type: {notify.nType}, Value: {notify.nValue}, "\
        f"timestamp:{notify.nTimeStamp}, msg: '{notify.sMsg.decode('utf-8')}', "\
        f"param1:{notify.nParam1}, param2:{notify.nParam2}, param3:{notify.nParam3}, param4:{notify.nParam4}")

# def main(argv):
#     serverIp = '10.1.1.198'

#     try:
#         opts, args = getopt.getopt(argv,"hs:",["server="])
#     except getopt.GetoptError:
#         print('NokovSDKClient.py -s <serverIp>')
#         sys.exit(2)

#     for opt, arg in opts:
#         if opt == '-h':
#             print('NokovSDKClient.py -s <serverIp>')
#             sys.exit()
#         elif opt in ("-s", "--server"):
#             serverIp = arg

#     print ('serverIp is %s' % serverIp)
#     print("Started the Nokov_SDK_Client Demo")
#     global client
#     client = PySDKClient()

#     ver = client.PyNokovVersion()
#     print('NokovSDK Sample Client 2.4.0.5428(NokovSDK ver. %d.%d.%d.%d)' % (ver[0], ver[1], ver[2], ver[3]))
#     print("Begin to init the SDK Client")
#     ret = client.Initialize(bytes(serverIp, encoding = "utf8"))
#     if ret == 0:
#         print("Connect to the Nokov Succeed")
#     else:
#         print("Connect Failed: [%d]" % ret)
#         exit(0)
    
#     print("\n1: Callback passive receiving data\n2: Host reading data\nEnter 1,2 select: ")
#     ch = input()
    
#     ret = client.PySetDataCallback(py_data_func, None)
#     if ret == 0:
#         print("py_data_func successfully registered")
#     else:
#         print(f"Failed to register py_data_func: {ret}")
    
#     if ch == 2:
#         t = Thread(target=read_data_func, args=(client,))
#         t.setDaemon(True)
#         t.start()
#     else:
#         client.PySetDataCallback(py_data_func, None)
#         client.PySetAnalogChFunc(py_analog_channel_func, None)

#     client.PySetVerbosityLevel(0)
#     client.PySetMessageCallback(py_msg_func)
#     client.PySetNotifyMsgCallback(py_notify_func, None)

#     serDes = ServerDescription()
#     client.PyGetServerDescription(serDes)

#     pdds = POINTER(DataDescriptions)()
#     client.PyGetDataDescriptions(pdds)
#     py_desc_func(pdds)

#     #Give 5 seconds to system to init forceplate device
    
#     ret = client.PyWaitForForcePlateInit(5000)
#     if (ret != 0):
#         print("Init ForcePlate Failed[%d]" % ret)
#         exit(0)

#     client.PySetForcePlateCallback(py_forcePlate_func, None)
#     while(input("Press q to quit\n") != "q"):
#         pass
 
# if __name__ == "__main__":
#    main(sys.argv[1:])
   


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

        
    
init_motion_capture()

print("\n1: Callback passive receiving data\n2: Host reading data\nEnter 1,2 select: ")


client.PySetDataCallback(py_data_func, None)
client.PySetVerbosityLevel(0)
client.PySetMessageCallback(py_msg_func)
client.PySetNotifyMsgCallback(py_notify_func, None)
serDes = ServerDescription()
client.PyGetServerDescription(serDes)
pdds = POINTER(DataDescriptions)()
client.PyGetDataDescriptions(pdds)
# py_desc_func(pdds)
ret = client.PyWaitForForcePlateInit(5000)
if (ret != 0):
    print("Init ForcePlate Failed[%d]" % ret)
    exit(0)

client.PySetForcePlateCallback(py_forcePlate_func, None)
while(input("Press q to quit\n") != "q"):
    pass