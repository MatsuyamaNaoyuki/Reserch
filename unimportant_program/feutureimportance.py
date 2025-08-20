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
from myclass.MyModel import ResNetGRUforshap
import japanize_matplotlib
from pathlib import Path
import shap





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




#変える部分-----------------------------------------------------------------------------------------------------------------

testloss = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\alluse_data32stride10\3d_testloss20250624_094326.pickle"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\mixhit_fortesttype.pickle"
motor_angle = True
motor_force = True
magsensor = True
L = 32
stride = 10

touch_vis = False
scatter_motor = False
row_data_swith = False




#-----------------------------------------------------------------------------------------------------------------

input_dim = 4 * motor_angle + 4 * motor_force + 9 * magsensor
output_dim = 12




pickle = detect_file_type(filename)
basepath = Path(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult")


if pickle:
    x_data,y_data , typedf= myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
else:
    x_data,y_data = myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
x_data = x_data.to(device)
y_data = y_data.to(device)

resultdir = os.path.dirname(testloss)
scaler_path = myfunction.find_pickle_files("scaler", resultdir)
scaler_data = myfunction.load_pickle(scaler_path)

minid = str(get_min_loss_epoch(testloss))
print(f"使用したephoch:{minid}")
modelpath = myfunction.find_pickle_files("epoch" + minid + "_", directory=resultdir, extension='.pth')
model = ResNetGRUforshap(input_dim=input_dim,output_dim=output_dim, hidden=128)
script_model = torch.jit.load(modelpath)
state_dict = script_model.state_dict() 

model.load_state_dict(state_dict)
model.to(device)




x_mean = torch.tensor(scaler_data['x_mean']).to(device)
x_std = torch.tensor(scaler_data['x_std']).to(device)
y_mean = torch.tensor(scaler_data['y_mean']).to(device)
y_std = torch.tensor(scaler_data['y_std']).to(device)
x_change = (x_data - x_mean) / x_std
y_change = (y_data - y_mean) / y_std

type_end_list = myfunction.get_type_change_end(typedf)


seq_x, seq_y, first_group_len= make_sequence_tensor_stride(x_change, y_data,type_end_list, L=L, stride=stride)


num_background = 50
indices = np.random.choice(len(seq_x), size=num_background, replace=False)
background = seq_x[indices]

num_explain = 50
indices_explain = np.random.choice(len(seq_x), size=num_explain, replace=False)
test_samples = seq_x[indices_explain]

torch.backends.cudnn.enabled = False
model.train()


explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_samples)
torch.backends.cudnn.enabled = True  # 終わったら元に戻す

shap.summary_plot(shap_values, test_samples)

# モデルのロード



# # x_data から 1 サンプルを取得（例: 0番目のサンプル）
# dis_array1 = []
# dis_array2 = []

# print(f"seq_x の長さ: {len(seq_x)}")
# print(f"first_group_len: {first_group_len}")
# # print(dis_array)


# start = time.time()
# for i in range(len(seq_x)):
#     sample_idx = i  # 推論したいサンプルのインデックス
#     single_sample = seq_x[sample_idx].unsqueeze(0)  # (input_dim,) -> (1, input_dim)
#     with torch.no_grad():  # 勾配計算を無効化
#         prediction = model_from_script(single_sample)
#     prediction = prediction * y_std + y_mean
    
#     distance = culc_gosa(prediction.tolist()[0], seq_y[sample_idx].tolist())
#     if i < first_group_len:
#         dis_array1.append(distance)
#     else:
#         dis_array2.append(distance)
# end = time.time()

# dis_array1 = np.array(dis_array1)
# dis_array2 = np.array(dis_array2)


# dis_array = np.concatenate([dis_array1, dis_array2], axis=0)
# column_means = np.mean(dis_array, axis=0)
# print("列ごとの平均:", column_means.round(2))



# print(end-start)
# # if touch_vis:
# #     make_touch_hist()

# # if scaler_data:
# #     make_scatter_plot_of_motor_and_error()

# # if row_data_swith:
# #     make_row_data_with_gosa(dis_array)