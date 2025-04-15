import time, datetime, getopt, sys
#resnetを実装したもの
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from myclass import myfunction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import random,pickle
import numpy as np
import pandas as pd
import os, sys, threading
from pathlib import Path
from myclass.MyDynamixel2 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
from nokov.nokovsdk import *

class NewThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        super().join(*args)
        return self._return

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

def culc_gosa(prediction, ydata):
    dis_array = np.zeros(4)
    # print(ydata)
    for i in range(4):
        pointpred =np.array([prediction[i * 3],prediction[i * 3 + 1],prediction[i * 3 + 2]])
        pointydata = np.array([ydata[3 * i], ydata[3 * i + 1], ydata[3 * i + 2]])
        distance = np.linalg.norm(pointpred - pointydata)
        dis_array[i] = distance
    return dis_array

def get_min_loss_epoch(file_path):
    testdf = pd.read_pickle(file_path)
    testdf = pd.DataFrame(testdf)
    filter_testdf = testdf[testdf.index % 10 == 0]
    minid = filter_testdf.idxmin()
    return minid.iloc[-1]

def detect_file_type(filename):
    # Pathオブジェクトで拡張子を取得
    file_extension = Path(filename).suffix.lower()

    # 拡張子に基づいてファイルタイプを判定
    if file_extension == '.pickle':
        return True
    elif file_extension == '.csv':
        return False
    else:
        return 'unknown'  # サポート外の拡張子

       
def get_dynamixel(Motors):
    motor_angle = Motors.get_present_angles()
    motor_current = Motors.get_present_currents()
    motor_data = myfunction.combine_lists(motor_angle, motor_current)



def get_magsensor(Ms, magpath):
    mag_data = Ms.get_value()


    

def move(Motors, howtomovepath):
    with open(howtomovepath, mode='br') as fi:
        change_angle = pickle.load(fi)
    len_angle = len(change_angle)
    print(change_angle)
    for i, angles in enumerate(change_angle):
        print(angles)
        print(str(i) + "/" +  str(len_angle))
        Motors.move_to_points(angles, times = 7)
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
            


def get_motioncapture(Ms, mcpath):
    frame = client.PyGetLastFrameOfMocapData()
    if frame :
                try:
                    motiondata = py_data_func(frame, client)
                finally:
                    client.PyNokovFreeFrame(frame)

        
        
class ResNetRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNetRegression, self).__init__()
        self.resnet = resnet18(weights=None)

        # 最初の畳み込み層を 2D Conv に変更
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        # BatchNorm2d を BatchNorm1d に変更する必要はありません
        self.resnet.bn1 = nn.BatchNorm2d(64)

        # 出力層を回帰問題用に変更
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)  # (batch_size, input_dim) -> (batch_size, 1, input_dim, 1)
        out = self.resnet(x)
        return out

input_dim = 4
output_dim = 12
learning_rate = 0.001
num_epochs = 300



model = ResNetRegression(input_dim=input_dim, output_dim=output_dim)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#変える部分-----------------------------------------------------------------------------------------------------------------

testloss = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech\angle_and_magsensor\3d_testloss20250217_224655.pickle"
howtomovepath = r"C"
motor_angle = True
motor_force = False
magsensor = True

#-----------------------------------------------------------------------------------------------------------------

basepath = Path(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult")






resultdir = os.path.dirname(testloss)
scaler_path = myfunction.find_pickle_files("scaler", resultdir)
scaler_data = myfunction.load_pickle(scaler_path)
x_mean = torch.tensor(scaler_data['x_mean']).to(device)
x_std = torch.tensor(scaler_data['x_std']).to(device)
y_mean = torch.tensor(scaler_data['y_mean']).to(device)
y_std = torch.tensor(scaler_data['y_std']).to(device)




#モデルのロード
minid = str(get_min_loss_epoch(testloss))
print(f"使用したephoch:{minid}")
modelpath = myfunction.find_pickle_files("epoch" + minid + "_", directory=resultdir, extension='.pth')
model_from_script = torch.jit.load(modelpath, map_location="cuda:0")
model_from_script.eval()



Motors = MyDynamixel()
Ms = MagneticSensor()


init_motion_capture()
Motors = MyDynamixel()
Ms = MagneticSensor()
results = {}
stop_event = threading.Event()

thread1 = threading.Thread(target=get_dynamixel, args=(Motors,), name="motor")
thread2 = threading.Thread(target=get_magsensor, args=(Ms,), name="magsensor")
thread3 = threading.Thread(target=move, args=(Motors,howtomovepath,), name="move")
thread4 = threading.Thread(target=get_motioncapture, args=(Ms,), name="motioncapture")





