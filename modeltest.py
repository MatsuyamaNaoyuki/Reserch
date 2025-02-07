import time
#resnetを実装したもの
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from myclass import myfunction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os, sys

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


# -------------------------------------------------------------------------------------


model_result_dir =r"sentan_morecam\all_use"
testlossname = "3d_testloss20250123_162722.pickle"
motor_angle = True
motor_force = True
magsensor = True
test_data_path =  r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\big\modifydata_big20250127_184801.pickle"




# -------------------------------------------------------------------------------------
base_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult"


base_path = os.path.join(base_path, model_result_dir)
testloss = os.path.join(base_path, testlossname)

minid = get_min_loss_epoch(testloss)

print(minid)
epochname = "epoch" + str(minid)

modelpath = myfunction.find_pickle_files(epochname, directory=base_path)




model_from_script = torch.jit.load(modelpath, map_location="cuda:0")

filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\big\modifydata_big20250127_184801.pickle"
x_data,y_data = myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
x_data = x_data.to(device)
y_data = y_data.to(device)
x_mean = x_data.mean()
x_std = x_data.std()
y_mean = y_data.mean()
y_std = y_data.std()
x_change = (x_data - x_data.mean()) / x_data.std()
y_change = (y_data - y_data.mean()) / y_data.std()

model_from_script.eval()

# x_data から 1 サンプルを取得（例: 0番目のサンプル）
dis_array = np.zeros((1000, 4))
# print(dis_array)
for i in range(1000):
    # sample_idx = random.randint(int(len(x_data) * 0.8 ),len(x_data)-1)  # 推論したいサンプルのインデックス
    sample_idx = random.randint(0,len(x_data)-1)  # 推論したいサンプルのインデックス
    single_sample = x_change[sample_idx].unsqueeze(0)  # (input_dim,) -> (1, input_dim)
    # 推論を行う（GPUが有効ならGPU上で実行）
    with torch.no_grad():  # 勾配計算を無効化
        prediction = model_from_script(single_sample)
    single_sample = single_sample * x_std + x_mean
    prediction = prediction * y_std + y_mean
    distance = culc_gosa(prediction.tolist()[0], y_data[sample_idx].tolist())
    dis_array[i, :] = distance

print(dis_array)
column_means = np.mean(dis_array, axis=0)
print("列ごとの平均:", column_means)