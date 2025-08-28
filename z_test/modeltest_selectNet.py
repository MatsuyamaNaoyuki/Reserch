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

MDNpath= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\MDN2\model.pth"
selectorpath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\selecttest\selector.pth"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortestnew.pickle"

touch_vis = True
scatter_motor = True
row_data_swith = True
#-----------------------------------------------------------------------------------------------------------------

input_dim = 3

basepath = Path(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult")



x_data, y_data, typedf = myfunction.read_pickle_to_torch(filename, motor_angle=True, motor_force=True, magsensor=True)
y_last3 = y_data[:, -3:]

mdn_scaler_path = myfunction.find_pickle_files("scaler", os.path.dirname(MDNpath))
mdn_scaler = myfunction.load_pickle(mdn_scaler_path)



select_scaler_path = myfunction.find_pickle_files("scaler", os.path.dirname(selectorpath))
select_scaler_data = myfunction.load_pickle(select_scaler_path)
sel_x_mean = torch.tensor(select_scaler_data['x_mean'],device = device).float()
sel_x_std = torch.tensor(select_scaler_data['x_std'],device = device).float()
sel_y_mean = torch.tensor(select_scaler_data['y_mean'],device = device).float()
sel_y_std = torch.tensor(select_scaler_data['y_std'],device = device).float()
sel_fitA = select_scaler_data['fitA']

mdn_x_mean = torch.tensor(mdn_scaler['x_mean'], device=device).float()
mdn_x_std = torch.tensor(mdn_scaler['x_std'], device=device).float()
mdn_y_mean = torch.tensor(mdn_scaler['y_mean'], device=device).float()
mdn_y_std = torch.tensor(mdn_scaler['y_std'], device=device).float()


alphaA = torch.ones_like(sel_fitA.amp)
xA_proc = Mydataset.apply_align_torch(x_data, sel_fitA, alphaA)
rot3_raw, force3_raw, m9_raw = torch.split(xA_proc, [3,3,9], dim=1)

t3_std_mdn = (rot3_raw.to(device) - mdn_x_mean) / mdn_x_std

m9_session_mean = m9_raw.mean(dim = 0, keepdim=True).to(device)
sel_m9_mean = sel_x_mean[6:15].unsqueeze(0)
m9_raw_shifted = m9_raw.to(device) + (sel_m9_mean - m9_session_mean)

m9_std_sel = (m9_raw_shifted - sel_m9_mean) / sel_x_std[6:15].unsqueeze(0)  # (N,9)




model_MDN = torch.jit.load(MDNpath, map_location="cuda:0")
model_MDN.eval()
model_select = torch.jit.load(selectorpath, map_location="cuda:0")
model_select.eval()



prediction_array = []


n_ok = 0; n = 0


with torch.no_grad():

    for i in range(t3_std_mdn.size(0)):
        t3 = t3_std_mdn[i:i+1]
        m9i = m9_std_sel[i:i+1]
        pi, mu_std_mdn, sigma = model_MDN(t3)
        mu_world = mu_std_mdn * mdn_y_std + mdn_y_mean
        mu_std_sel = (mu_world - sel_y_mean) / sel_y_std
        feats = torch.cat([mu_std_sel[:,0,:], mu_std_sel[:,1,:], m9i], dim=1)  # (1,15)
        logits, _ = model_select(feats)      # (1,2)
        pred_idx = logits.argmax(dim=1).item()

        # 選ばれたμは“実スケール”で返す（可視化や保存はこちらが正しい）
        pred_world = mu_world[0, pred_idx, :]            # (3,)
        prediction_array.append(pred_world.detach().cpu())






myfunction.print_val(prediction_array)
myfunction.wirte_pkl(prediction_array, "result")


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