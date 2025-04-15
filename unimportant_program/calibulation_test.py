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
from pathlib import Path
import math

torch.set_printoptions(sci_mode=False)
def culc_gosa(prediction, ydata):
    dis_array = np.zeros(4)
    # print(ydata)
    for i in range(4):
        pointpred =np.array([prediction[i * 3],prediction[i * 3 + 1],prediction[i * 3 + 2]])
        pointydata = np.array([ydata[3 * i], ydata[3 * i + 1], ydata[3 * i + 2]])
        distance = np.linalg.norm(pointpred - pointydata)
        dis_array[i] = distance
    return dis_array


def culc_gosa_total(prediction, ydata):
    dis_array = np.zeros(4)
    # print(ydata)
    for i in range(4):
        pointpred =np.array([prediction[i * 3],prediction[i * 3 + 1],prediction[i * 3 + 2]])
        pointydata = np.array([ydata[3 * i], ydata[3 * i + 1], ydata[3 * i + 2]])
        distance = np.linalg.norm(pointpred - pointydata)
        dis_array[i] = distance
    total = sum(dis_array)
    return total



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

testloss = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\currentOK0203\correctmean_norandom\3d_testloss20250205_043927.pickle"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\currentOK0203\currentOKtest_020320250203_201037.pickle"
trainpath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\currentOK0203\currentOK20250203_180002.pickle"
motor_angle = True
motor_force = True
magsensor = True





#-----------------------------------------------------------------------------------------------------------------
pickle = detect_file_type(filename)
testloss = Path(testloss)
if pickle:
    x_data,y_data = myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
else:
    x_data,y_data = myfunction.read_csv_to_torch(filename, motor_angle, motor_force, magsensor)
x_data = x_data.to(device)
y_data = y_data.to(device)



pickle = detect_file_type(trainpath)

if pickle:
    train_x_data,train_y_data = myfunction.read_pickle_to_torch(trainpath, motor_angle, motor_force, magsensor)
else:
    train_x_data,train_y_data = myfunction.read_csv_to_torch(trainpath, motor_angle, motor_force, magsensor)
train_x_data = train_x_data.to(device)
train_y_data = train_y_data.to(device)


max_index = torch.argmax(x_data[:, 0])
max_yrow = y_data[max_index]
max_xrow = x_data[max_index]


# min_gosa = math.inf
# max_yrow = max_yrow.view(12, 1) 
# max_yrow = max_yrow.view(4,3)
# for i, train_row in enumerate(train_y_data):
#     train_row = train_row.view(4,3)
#     distances = torch.norm(max_yrow - train_row, dim=1)
#     gosa = torch.sum(distances)
#     if gosa < min_gosa:
#         neatist_train_y = train_row
#         nearist_index = i
#         min_gosa = gosa  

# print(f"max_row = {max_yrow}, nearist_train_y = {neatist_train_y}")
# print(f"min_gosa = {min_gosa}")
# print(f"nearist_index = {nearist_index}")
nearist_index = 28229
neatist_train_x = train_x_data[nearist_index]


diff_base = max_xrow - neatist_train_x


# x_data = x_data - diff_base
print(max_xrow)
print(neatist_train_x)

# print(f"x_data[max_index] = {x_data[max_index]}")
# print(f"x_data[nearidt] = {train_x_data[nearist_index]}")
resultdir = os.path.dirname(testloss)
scaler_path = myfunction.find_pickle_files("scaler", resultdir)
scaler_data = myfunction.load_pickle(scaler_path)
x_mean = train_x_data.mean(dim=0, keepdim=True)
x_std = train_x_data.std(dim=0, keepdim=True)
y_mean = train_y_data.mean(dim=0, keepdim=True)
y_std = train_y_data.std(dim=0, keepdim=True)
# x_mean = torch.tensor(scaler_data['x_mean']).to(device)
# x_std = torch.tensor(scaler_data['x_std']).to(device)
# y_mean = torch.tensor(scaler_data['y_mean']).to(device)
# y_std = torch.tensor(scaler_data['y_std']).to(device)
x_change = (x_data - x_mean) / x_std
y_change = (y_data - y_mean) / y_std


print(f"x_change[max_index] = {x_change[max_index]}")

#モデルのロード
minid = str(get_min_loss_epoch(testloss))
print(f"使用したephoch:{minid}")
modelpath = myfunction.find_pickle_files(minid, directory=resultdir, extension='.pth')
model_from_script = torch.jit.load(modelpath, map_location="cuda:0")
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
column_var = np.var(dis_array, axis = 0)
print("列ごとの平均:", column_means)
print("列ごとの分散:", column_var)


