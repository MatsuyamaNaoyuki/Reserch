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
import japanize_matplotlib
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

def make_sequence_tensor_stride(x, y, typedf, L, stride):
    typedf = typedf.tolist()
    typedf.insert(0, 0)
    total_span = (L - 1) * stride

    seq_x, seq_y = [], []

    nan_mask = torch.isnan(x).any(dim=1)
    nan_rows = nan_mask.nonzero(as_tuple=True)[0].tolist()

    nan_rows_set = set(nan_rows)  # 高速化のため set にしておく
    first_group_len = None  # ← 追加：最初のグループの shape[0] を格納

    for i in range(len(typedf) - 1):
        start = typedf[i] + total_span
        end = typedf[i + 1]
        if end <= start:
            continue

        js = torch.arange(start, end, device=x.device)
        relative_indices = torch.arange(L-1, -1, -1, device=x.device) * stride
        indices = js.unsqueeze(1) - relative_indices  # shape: (num_seq, L)

       # --- ここで NaN 系列を除外する ---
        # indices を CPU に移動して numpy に変換
        indices_np = indices.cpu().numpy()
        # nan_rows が含まれているか判定
        valid_mask = []
        for row in indices_np:
            if any(idx in nan_rows_set for idx in row):
                valid_mask.append(False)  # nan を含む → 無効
            else:
                valid_mask.append(True)   # nan を含まない → 有効

        valid_mask = torch.tensor(valid_mask, device=x.device)

        # 有効な indices だけ残す
        indices = indices[valid_mask]
        js = js[valid_mask]

        if indices.shape[0] == 0:
            continue  # 有効な系列がなければスキップ
        # 最初のグループだけ取得
        if first_group_len is None:
            first_group_len = indices.shape[0]

        x_seq = x[indices]
        y_seq = y[js]

        seq_x.append(x_seq)
        seq_y.append(y_seq)

    seq_x = torch.cat(seq_x, dim=0)
    seq_y = torch.cat(seq_y, dim=0)

    return seq_x, seq_y, first_group_len  # ← 追加：最初のグループ長も返す


def make_touch_hist():
    point4_array1 = dis_array1[:, 3]
    point4_array2 = dis_array2[:, 3]

    # ヒストグラム描画
    plt.figure(figsize=(8, 5))
    plt.hist(point4_array1, bins='auto', alpha=0.6, label='dis_array1 (first part)', edgecolor='black')
    plt.hist(point4_array2, bins='auto', alpha=0.6, label='dis_array2 (second part)', edgecolor='black')

    # ラベルとタイトル
    plt.xlabel('Distance Error (Point 4)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Point 4 Errors: dis_array1 vs dis_array2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def to_numpy_safe(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    else:
        return np.asarray(t)


def make_scatter_plot_of_motor_and_error():
    motorsum = []
    for i in range(len(seq_x)):
        last_step = seq_x[i][-1]            # shape: (17,)
        last_step = last_step * x_std + x_mean
        motorsum.append(last_step[0:4].sum().item())

    motorsum = np.array(motorsum)           # (N,)

    # dis_array1, dis_array2 はすでに NumPy 配列 (N1,4), (N2,4)
    motorsum1 = motorsum[:first_group_len]          # (N1,)
    motorsum2 = motorsum[first_group_len:]          # (N2,)

    # -----------------------------------------------------------
    # 2. 散布図を 2 色で描画
    # -----------------------------------------------------------
    plt.figure(figsize=(8, 5))

    # グループ1（dis_array1）の Point-4 誤差
    plt.scatter(motorsum1, dis_array1[:, 3],
                alpha=0.7, label='接触あり')

    # グループ2（dis_array2）の Point-4 誤差
    plt.scatter(motorsum2, dis_array2[:, 3],
                alpha=0.7, label='接触なし')

    plt.xlabel('Motor Sum')
    plt.ylabel('Distance Error (Point 4)')
    plt.title('Scatter: Motor Sum vs Point-4 Error (2 Groups)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def make_row_data_with_gosa(dis_array):
    row_data = []
    for i in range(len(seq_x)):
        last_step = seq_x[i][-1]            # shape: (17,)
        last_step = last_step * x_std + x_mean
        row_data.append(last_step.tolist()[0])
    
    
    rowdata = np.array(row_data)
    dis_array = np.array(dis_array[:, 3])
    mean_val = dis_array.mean()

    # ==== プロット準備 ====
    plt.figure(figsize=(8, 5))
    x = list(range(len(rowdata)))  # 行番号

    # ==== 背景色（縦帯）を行ごとに塗る ====
    for i in range(len(rowdata)):
        color = 'blue' if dis_array[i] < mean_val else 'red'
        plt.axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=0.3)

    # ==== 各列ごとの折れ線（列を固定して描く）====
    for col in [0,1,2,3]:  # 0～3列目
        y = rowdata[:, col]
        plt.plot(x, y, marker='o', markersize=4, linewidth=1.0, label=f'Column {col}')

    # ==== 装飾 ====
    plt.xlabel('Row Index')
    plt.ylabel('磁気センサー')
    plt.title('磁気センサーと誤差の関係')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#変える部分-----------------------------------------------------------------------------------------------------------------

modelpath= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\allusenan\model20250715_203305.pth"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\mixhit_fortesttype.pickle"
motor_angle = True
motor_force = True
magsensor = True
L = 32
stride = 1

touch_vis = True
scatter_motor = True
row_data_swith = True
#-----------------------------------------------------------------------------------------------------------------

input_dim = 3 * motor_angle + 3 * motor_force + 9 * magsensor
output_dim = 12




pickle = detect_file_type(filename)
basepath = Path(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult")


if pickle:
    x_data,y_data , typedf= myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
else:
    x_data,y_data = myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
x_data = x_data.to(device)
y_data = y_data.to(device)




resultdir = os.path.dirname(modelpath)
scaler_path = myfunction.find_pickle_files("scaler", resultdir)
scaler_data = myfunction.load_pickle(scaler_path)
x_mean = torch.tensor(scaler_data['x_mean']).to(device)
x_std = torch.tensor(scaler_data['x_std']).to(device)
y_mean = torch.tensor(scaler_data['y_mean']).to(device)
y_std = torch.tensor(scaler_data['y_std']).to(device)
x_change = (x_data - x_mean) / x_std
y_change = (y_data - y_mean) / y_std

type_end_list = myfunction.get_type_change_end(typedf)


seq_x, seq_y, first_group_len= make_sequence_tensor_stride(x_change, y_data,type_end_list, L=L, stride=stride)



# モデルのロード

model_from_script = torch.jit.load(modelpath, map_location="cuda:0")
model_from_script.eval()



# x_data から 1 サンプルを取得（例: 0番目のサンプル）
dis_array1 = []
dis_array2 = []
prediction_array = []
real_array = []
print(f"seq_x の長さ: {len(seq_x)}")
print(f"first_group_len: {first_group_len}")
# print(dis_array)


start = time.time()
for i in range(len(seq_x)):
    sample_idx = i  # 推論したいサンプルのインデックス
    single_sample = seq_x[sample_idx].unsqueeze(0)  # (input_dim,) -> (1, input_dim)
    with torch.no_grad():  # 勾配計算を無効化
        prediction = model_from_script(single_sample)
    prediction = prediction * y_std + y_mean
    prediction_array.append(prediction)
    real_array.append(seq_y[sample_idx].tolist())
    distance = culc_gosa(prediction.tolist()[0], seq_y[sample_idx].tolist())
    if i < first_group_len:
        dis_array1.append(distance)
    else:
        dis_array2.append(distance)
end = time.time()

dis_array1 = np.array(dis_array1)
dis_array2 = np.array(dis_array2)


dis_array = np.concatenate([dis_array1, dis_array2], axis=0)
column_means = np.mean(dis_array, axis=0)
column_means1 = np.mean(dis_array1, axis=0)
column_means2 = np.mean(dis_array2, axis=0)
print("列ごとの平均:", column_means.round(2))
print("1列ごとの平均:", column_means1.round(2))
print("2列ごとの平均:", column_means2.round(2))
# myfunction.send_message_for_test(column_means.round(2))


myfunction.wirte_pkl(prediction_array, r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\allusenan\result")
myfunction.wirte_pkl(real_array, r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\allusenan\real")
print(end-start)
if touch_vis:
    make_touch_hist()

if scaler_data:
    make_scatter_plot_of_motor_and_error()

if row_data_swith:
    make_row_data_with_gosa(dis_array)