import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel2 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
import numpy as np
from nokov.nokovsdk import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    return motiondata




def get_dynamixel(Motors):
    # now_time  = datetime.datetime.now()
    motor_angle = Motors.get_present_angles()
    motor_current = Motors.get_present_currents()
    motor_data = myfunction.combine_lists(motor_angle, motor_current)
    # motor_data.insert(0, now_time)
    return motor_data

def mag_data_change2(row):
    split_value = row.split('/')
    if len(split_value) != 9:
        split_value = split_value[1:]
    int_list = [int(x) for x in split_value]
    return int_list


def get_magsensor(Ms):
    mag_data = Ms.get_value()
    return mag_data


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

def get_estimate_coordinate(Motors, Ms, x_mean, x_std, y_mean, y_std):
    motordata = get_dynamixel(Motors)
    magdata = get_magsensor(Ms)
    mag = mag_data_change2(magdata)
    margedata = motordata + mag
    np_x_value= np.array(margedata).astype("float32") 
    tensor_data_x = torch.tensor(np_x_value, dtype=torch.float32)
    tensor_data_x = tensor_data_x.to(device)
    x_change = (tensor_data_x - x_mean) / x_std
    # 推論を行う（GPUが有効ならGPU上で実行）
    with torch.no_grad():  # 勾配計算を無効化
        prediction = model_from_script(x_change)
    prediction = prediction * y_std + y_mean                                                                    
    return prediction

def culc_gosa(prediction, ydata):
    dis_array = np.zeros(4)
    # print(ydata)
    for i in range(4):
        pointpred =np.array([prediction[i * 3],prediction[i * 3 + 1],prediction[i * 3 + 2]])
        pointydata = np.array([ydata[3 * i], ydata[3 * i + 1], ydata[3 * i + 2]])
        distance = np.linalg.norm(pointpred - pointydata)
        dis_array[i] = distance
    return dis_array



#変える部分-----------------------------------------------------------------------------------------------------------------

testloss = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger_mixhit\alluse\3d_testloss20250414_143728.pickle"
motor_angle = True
motor_force = True
magsensor = True

testin = None
#-----------------------------------------------------------------------------------------------------------------

input_dim = 4 * motor_angle + 4 * motor_force + 9 * magsensor
output_dim = 12
learning_rate = 0.001
num_epochs = 300

model = ResNetRegression(input_dim=input_dim, output_dim=output_dim)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

resultdir = os.path.dirname(testloss)
scaler_path = myfunction.find_pickle_files("scaler", resultdir)
scaler_data = myfunction.load_pickle(scaler_path)
x_mean = torch.tensor(scaler_data['x_mean']).to(device)
x_std = torch.tensor(scaler_data['x_std']).to(device)
y_mean = torch.tensor(scaler_data['y_mean']).to(device)
y_std = torch.tensor(scaler_data['y_std']).to(device)





#モデルのロード
minid = str(myfunction.get_min_loss_epoch(testloss))
print(f"使用したephoch:{minid}")
modelpath = myfunction.find_pickle_files("epoch" + minid + "_", directory=resultdir, extension='.pth')
model_from_script = torch.jit.load(modelpath, map_location="cuda:0")
model_from_script.eval()


init_motion_capture()
Motors = MyDynamixel()
Ms = MagneticSensor()
estimate_cordinate = get_estimate_coordinate(Motors, Ms, x_mean, x_std, y_mean, y_std)
real_cordinate = get_motioncapture()

distance = culc_gosa(estimate_cordinate.tolist(), real_cordinate.tolist())
print(distance)


