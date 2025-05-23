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


def make_sequence_tensor_stride(x, y, L=6, stride=1):
    """
    過去Lステップをstride間隔で取り出す（飛ばし取り）時系列データセットを作成

    Parameters:
        x (Tensor): 入力データ（[N, D]）
        y (Tensor): 出力データ（[N, ?]）
        L (int): 時系列の長さ
        stride (int): 間隔（n個飛ばし）
    """
    seq_x, seq_y = [], []
    total_span = (L - 1) * stride  # 必要な履歴全体の長さ

    for i in range(total_span, len(x)):
        indices = [i - j * stride for j in reversed(range(L))]  # 取り出すインデックス
        seq_x.append(x[indices])
        seq_y.append(y[i])
    
    return torch.stack(seq_x), torch.stack(seq_y)




#変える部分-----------------------------------------------------------------------------------------------------------------

testloss = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\data30stride2\3d_testloss20250520_074333.pickle"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\mixhit_fortest20250227_135315.pickle"
motor_angle = True
motor_force = True
magsensor = True
L = 30
stride = 2
testin = None
#-----------------------------------------------------------------------------------------------------------------

input_dim = 4 * motor_angle + 4 * motor_force + 9 * magsensor
output_dim = 12




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


seq_x, seq_y = make_sequence_tensor_stride(x_change, y_data, L=L, stride=stride)
print(f"type:{type(seq_x)}")
print(f"seq_x.shape{seq_x.shape}")  # 入力テンソルの shape: (N, L, D)
print(f"seq_y.shape{seq_y.shape}")  # 出力テンソルの shape: (N, D_out)