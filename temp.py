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



def make_sequence_tensor_stride(x, y, typedf, L, stride):
    typedf = typedf.tolist()
    typedf.insert(0, 0)
    total_span = (L - 1) * stride

    seq_x, seq_y = [], []

    nan_mask = torch.isnan(x).any(dim=1)
    nan_rows = nan_mask.nonzero(as_tuple=True)[0].tolist()

    nan_rows_set = set(nan_rows)  # 高速化のため set にしておく
    first_group_len = None  # ← 追加：最初のグループの shape[0] を格納

    jslist = []

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
        jslist.append(js)

    seq_x = torch.cat(seq_x, dim=0)
    seq_y = torch.cat(seq_y, dim=0)
    jslist = torch.cat(jslist, dim = 0)
    return seq_x, seq_y, first_group_len ,jslist # ← 追加：最初のグループ長も返す



#変える部分-----------------------------------------------------------------------------------------------------------------

modelpath= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\GRUseikika\model.pth"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortestnew.pickle"
basefilepath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit1500kaifortrain.pickle"
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





basepath = Path(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult")



x_data,y_data , typedf= myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
base_x_data ,base_y_data, _ = myfunction.read_pickle_to_torch(basefilepath, motor_angle, motor_force, magsensor)






base_x_data_mean = base_x_data.nanmean(dim=0)
# x_data_mean = x_data.nanmean(dim=0)


x_data = x_data - x_data[0] + base_x_data[0]

x_data = x_data.to(device)
y_data = y_data.to(device)

resultdir = os.path.dirname(modelpath)
scaler_path = myfunction.find_pickle_files("scaler", resultdir)
scaler_data = myfunction.load_pickle(scaler_path)
x_min = torch.tensor(scaler_data['x_min']).to(device)
x_max = torch.tensor(scaler_data['x_max']).to(device)
y_min = torch.tensor(scaler_data['y_min']).to(device)
y_max = torch.tensor(scaler_data['y_max']).to(device)
x_scale = (x_max - x_min).clamp(min=1e-8)
y_scale = (y_max - y_min).clamp(min=1e-8)

x_change = (x_data - x_min) / x_scale
y_change = (y_data - y_min) / y_scale





type_end_list = myfunction.get_type_change_end(typedf)


seq_x, seq_y, first_group_len, js= make_sequence_tensor_stride(x_change, y_data,type_end_list, L=L, stride=stride)




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
# for i in range(6000):
    sample_idx = i  # 推論したいサンプルのインデックス
    single_sample = seq_x[sample_idx].unsqueeze(0)  # (input_dim,) -> (1, input_dim)
    with torch.no_grad():  # 勾配計算を無効化
        prediction = model_from_script(single_sample)
    prediction = prediction * y_scale + y_min
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


# dis_array = np.concatenate([dis_array1, dis_array2], axis=0)
# column_means = np.mean(dis_array, axis=0)
# column_means1 = np.mean(dis_array1, axis=0)
# column_means2 = np.mean(dis_array2, axis=0)
# print("列ごとの平均:", column_means.round(2))
# print("1列ごとの平均:", column_means1.round(2))
# print("2列ごとの平均:", column_means2.round(2))
# myfunction.send_message_for_test(column_means.round(2))

parent = os.path.dirname(modelpath)
resultpath = os.path.join(parent, "result") 
myfunction.wirte_pkl(prediction_array, resultpath)
js_path = os.path.join(parent, "js") 
myfunction.wirte_pkl(js, js_path)

# myfunction.wirte_pkl(prediction_array, r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\allusenan\result")
# myfunction.wirte_pkl(real_array, r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\allusenan\real")
print(end-start)
