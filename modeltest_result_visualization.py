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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

testloss = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger_mixhit\alluse\3d_testloss20250414_143728.pickle"
filename = r"C:\Users\WRS\Desktop\Matsuyama\Reserch\tube_softfinger_test_10_20250415_172035.pickle"
motor_angle = True
motor_force = True
magsensor = True

testin = None
#-----------------------------------------------------------------------------------------------------------------
pickle = detect_file_type(filename)
basepath = Path(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult")


if pickle:
    x_data,y_data = myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
else:
    x_data,y_data = myfunction.read_csv_to_torch(filename, motor_angle, motor_force, magsensor)
x_data = x_data.to(device)
y_data = y_data.to(device)


if testin is not None:
    test_indices = myfunction.load_pickle(testin)
    x_data = x_data[test_indices]  
    y_data = y_data[test_indices]


resultdir = os.path.dirname(testloss)
scaler_path = myfunction.find_pickle_files("scaler", resultdir)
scaler_data = myfunction.load_pickle(scaler_path)
x_mean = torch.tensor(scaler_data['x_mean']).to(device)
x_std = torch.tensor(scaler_data['x_std']).to(device)
y_mean = torch.tensor(scaler_data['y_mean']).to(device)
y_std = torch.tensor(scaler_data['y_std']).to(device)
x_change = (x_data - x_mean) / x_std
y_change = (y_data - y_mean) / y_std




#モデルのロード
minid = str(get_min_loss_epoch(testloss))
print(f"使用したephoch:{minid}")
modelpath = myfunction.find_pickle_files("epoch" + minid + "_", directory=resultdir, extension='.pth')
model_from_script = torch.jit.load(modelpath, map_location="cuda:0")
model_from_script.eval()


#可視化される点のインデックス
visual_index = range(0,len(x_data),10)

estimation_array = np.zeros((len(visual_index), 12))
dis_array=np.zeros((len(visual_index), 4))

for i, sample_idx in enumerate(visual_index):
    single_sample = x_change[sample_idx].unsqueeze(0)  # (input_dim,) -> (1, input_dim)
    with torch.no_grad():  # 勾配計算を無効化
        prediction = model_from_script(single_sample)
    single_sample = single_sample * x_std + x_mean
    prediction = prediction * y_std + y_mean
    estimation_array[i, :] = prediction.tolist()[0]
    distance = culc_gosa(prediction.tolist()[0], y_data[sample_idx].tolist())
    dis_array[i, :] = distance
    

list_ydata = y_data.tolist()
list_ydata = [list_ydata[i] for i in visual_index]


x = [row[9] for row in list_ydata]
y = [row[10] for row in list_ydata]
z = [row[11] for row in list_ydata]
ex = [row[9] for row in estimation_array]
ey = [row[10] for row in estimation_array]
ez = [row[11] for row in estimation_array]

vx = [ai - bi for ai, bi in zip(ex, x)]
vy = [ai - bi for ai, bi in zip(ey, y)]
vz = [ai - bi for ai, bi in zip(ez, z)]

magnitude = np.sqrt(np.array(vx)**2 + np.array(vy)**2 + np.array(vz)**2)

# カラーマップを作成
norm = plt.Normalize(magnitude.min(), magnitude.max())
colors = cm.viridis(norm(magnitude))
print(colors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 点群をプロット
ax.scatter(x, y, z, marker='.', color=colors)
ax.scatter(ex, ey, ez, marker='.', color=colors)
ax.quiver(x, y, z, vx, vy, vz, length=1, normalize=False, color=colors,arrow_length_ratio=0.5)
# ラベル付け
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

max_range = max(
    max(x) - min(x),
    max(y) - min(y),
    max(z) - min(z),
    max(ex) - min(ex),
    max(ey) - min(ey),
    max(ez) - min(ez)
) / 2.0

mid_x = (max(x) + min(x)) / 2.0
mid_y = (max(y) + min(y)) / 2.0
mid_z = (max(z) + min(z)) / 2.0

ax.set_box_aspect([1, 1, 1])  # 各軸の比率を同じに設定
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
# 表示
plt.show()