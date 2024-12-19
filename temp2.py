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

input_dim = 17
output_dim = 12
learning_rate = 0.001
num_epochs = 300



model = ResNetRegression(input_dim=input_dim, output_dim=output_dim)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# testloss = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\alldata_tewntydata\\modi_margeMc_alldata_tewntydata_testloss20241218_024252.pickle"

# minid = get_min_loss_epoch(testloss)

# print(minid)


modelpath = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\alldata_tewntydata\\modi_margeMc_alldata_tewntydata_model_epoch430_20241218_024130.pth"

model_from_script = torch.jit.load(modelpath, map_location="cuda:0")

filename = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\dataset_margemag_tewnty.csv"
x_data,y_data = myfunction.read_csv_to_torch(filename)
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
dis_array = np.zeros((100, 12))
# print(dis_array)
for i in range(100):
    sample_idx = i # 推論したいサンプルのインデックス
    single_sample = x_change[sample_idx].unsqueeze(0)  # (input_dim,) -> (1, input_dim)
    # 推論を行う（GPUが有効ならGPU上で実行）
    with torch.no_grad():  # 勾配計算を無効化
        prediction = model_from_script(single_sample)
    single_sample = single_sample * x_std + x_mean
    prediction = prediction * y_std + y_mean
    prediction = prediction.tolist()[0]
    dis_array[i, :] = prediction

print(type(dis_array))
df = pd.DataFrame(dis_array, columns=["Mc2x", "Mc2y", "Mc2z", "Mc3x", "Mc3y", "Mc3z","Mc4x", "Mc4y", "Mc4z","Mc5x", "Mc5y", "Mc5z"])

# CSV ファイルに書き出し
df.to_csv("C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\output.csv", index=False)