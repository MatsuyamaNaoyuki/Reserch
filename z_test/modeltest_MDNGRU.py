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
from myclass import Mydataset





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

def save_checkpoint(epoch, model, optimizer, record_train_loss, record_test_loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'record_train_loss': record_train_loss,
        'record_test_loss': record_test_loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}.")


def build_motor_seq(rot3_std, type_end_list, L=16, stride=1):
    rot3_std = rot3_std.to(device)
    type_end_list = type_end_list.tolist()
    type_end_list.insert(0, 0)
    total_span = (L - 1) * stride

    seq_rot, js_all = [], []
    nan_mask = torch.isnan(rot3_std).any(dim=1).to(device)

    for i in range(len(type_end_list)-1):

        start = type_end_list[i] + total_span
        end   = type_end_list[i+1]
        if end <= start:
            continue

        js = torch.arange(start, end, device=device)  # 現在時刻の位置
        # 過去Lフレームのインデックスを作る
        rel = torch.arange(L-1, -1, -1, device=device) * stride   
         
        idx = js.unsqueeze(1) - rel    # [num, L]
        # ここで NaN を含む行を除外（必要なら）

        valid_mask = ~nan_mask[idx].any(dim=1)


        idx = idx[valid_mask]
        js = js[valid_mask]


        seq_rot.append(rot3_std[idx])  # [num, L, 9]
        js_all.append(js)

    seq_rot = torch.cat(seq_rot, dim=0) if len(seq_rot)>0 else torch.empty(0, L, 3, device=device)
    js_all  = torch.cat(js_all, dim=0)  if len(js_all)>0  else torch.empty(0, device=device, dtype=torch.long)

    # rot3_std/js_all からMDNを呼び、y_std/js_all を教師に使うため、js_allも返す
    return seq_rot, js_all



#変える部分-----------------------------------------------------------------------------------------------------------------

modelpath= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\re3tubefinger0912\MDNGRU\model.pth"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\re3tubefinger0912\mixhit10kaifortest.pickle"
L = 32
stride = 1
touch_vis = True
scatter_motor = True
row_data_swith = True
seiki = True
kijun = False
#-----------------------------------------------------------------------------------------------------------------

input_dim = 3




pickle = detect_file_type(filename)
basepath = Path(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult")

rotate_data, y_data, typedf= myfunction.read_pickle_to_torch(filename, True, False, False)
y_last3 = y_data[:, -3:]


rotate_data = rotate_data.to(device)
y_last3 = y_last3.to(device)


resultdir = os.path.dirname(modelpath)
scaler_path = myfunction.find_pickle_files("scaler", resultdir)
scaler_data = myfunction.load_pickle(scaler_path)
if seiki:
    x_min = torch.tensor(scaler_data['x_min'], device=device).float()
    x_max = torch.tensor(scaler_data['x_max'], device=device).float()
    y_min = torch.tensor(scaler_data['y_min'], device=device).float()
    y_max = torch.tensor(scaler_data['y_max'], device=device).float()
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    x_change = (rotate_data - x_min) /x_scale
    y_change = (y_last3 - y_min) /y_scale
else:
    x_mean = torch.tensor(scaler_data['x_mean'])
    x_std = torch.tensor(scaler_data['x_std'])
    y_mean = torch.tensor(scaler_data['y_mean'])
    y_std = torch.tensor(scaler_data['y_std'])
    fitA = scaler_data['fitA']

    alphaA = torch.ones_like(fitA.amp)

    xA_proc = Mydataset.apply_align_torch(rotate_data, fitA, alphaA)
    x_change = Mydataset.apply_standardize_torch(xA_proc, x_mean, x_std)
    y_change = Mydataset.apply_standardize_torch(y_last3, y_mean, y_std)
    y_std = y_std.to(device)
    y_mean = y_mean.to(device)




mask = torch.isfinite(x_change).all(dim=1) & torch.isfinite(y_last3).all(dim=1)
rotate_data_clean = x_change[mask]
y_last3_clean     = y_last3[mask]
y_last3_clean = y_last3_clean.to(device)

myfunction.print_val(rotate_data_clean.size())
type_end_list = myfunction.get_type_change_end(typedf)
rot_seq , js= build_motor_seq(rotate_data_clean, type_end_list, L=L, stride=stride)
y_last3_clean = y_last3_clean[js]
# rotate_data_clean ,force3_raw, m9_raw = torch.split(rotate_data_clean, [3,3,9], dim=1)

model_from_script = torch.jit.load(modelpath, map_location="cuda:0")
model_from_script.eval()



# x_data から 1 サンプルを取得（例: 0番目のサンプル）
dis_array = []
prediction_array = []
real_array = []
mu_list = []


for i in range(len(rot_seq)):

# for i in range(10000):
    sample_idx = i # 推論したいサンプルのインデックス
    single_sample = rot_seq[sample_idx].unsqueeze(0)  # (input_dim,) -> (1, input_dim)
    single_sample = single_sample.to(device)

    with torch.no_grad():  # 勾配計算を無効化
        pi, mu, sigma = model_from_script(single_sample)
    if seiki:
        mu = mu * y_scale + y_min
    else:    
        mu = mu * y_std + y_mean

    dis1 = torch.norm(mu[0,0, :] - y_last3_clean[i])
    dis2 = torch.norm(mu[0,1, :] - y_last3_clean[i])
    dis = torch.min(dis1, dis2)
    dis_array.append(dis)
    prediction_array.append(mu[0])


distance = sum(dis_array) / len(dis_array)

myfunction.print_val(distance)


dis_array = [v.item() for v in dis_array]
plt.plot(dis_array)  # marker="o" で点を丸で表示
plt.title("List as Line Plot")
plt.xlabel("Index")   # 横軸（0,1,2,...のインデックス）
plt.ylabel("Value")   # 縦軸（リストの値）
plt.grid(True)        # グリッド線を表示
plt.show()
parent = os.path.dirname(modelpath)
resultpath = os.path.join(parent, "result") 
myfunction.wirte_pkl(prediction_array, resultpath)
js_path = os.path.join(parent, "js") 
myfunction.wirte_pkl(js, js_path)

# dis_array1 = np.array(dis_array1)
# dis_array2 = np.array(dis_array2)


# dis_array = np.concatenate([dis_array1, dis_array2], axis=0)
# column_means = np.mean(dis_array, axis=0)
# column_means1 = np.mean(dis_array1, axis=0)
# column_means2 = np.mean(dis_array2, axis=0)
# print("列ごとの平均:", column_means.round(2))
# print("1列ごとの平均:", column_means1.round(2))
# print("2列ごとの平均:", column_means2.round(2))
# # myfunction.send_message_for_test(column_means.round(2))


# # myfunction.wirte_pkl(prediction_array, "result")
# myfunction.wirte_pkl(real_array, "real")
# print(end-start)
# if touch_vis:
#     make_touch_hist()

# if scaler_data:
#     make_scatter_plot_of_motor_and_error()

# if row_data_swith:
#     make_row_data_with_gosa(dis_array)