import time
import sys
sys.path.append('.')
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
from matplotlib.animation import FuncAnimation





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


def make_sequence_tensor_stride(x, y, typedf,L, stride):
    typedf = typedf.tolist()
    typedf.insert(0, 0)
    total_span = (L - 1) * stride

    seq_x, seq_y = [], []

    for i in range(len(typedf) - 1):
        start = typedf[i] + total_span
        end = typedf[i + 1]
        if end <= start:
            continue

        # j の全体リスト
        js = torch.arange(start, end, device=x.device)

        # indices テンソルをまとめて作成：(len(js), L)
        relative_indices = torch.arange(L-1, -1, -1, device=x.device) * stride
        indices = js.unsqueeze(1) - relative_indices  # shape: (num_seq, L)

        # <<< ここで indices を表示 >>>
        print(f"[Group {i}] indices shape: {indices.shape}")
        print(indices)

        # x と y を一括取得
        x_seq = x[indices]  # shape: (num_seq, L, D)
        y_seq = y[js]       # shape: (num_seq, D_out)

        seq_x.append(x_seq)
        seq_y.append(y_seq)

    seq_x = torch.cat(seq_x, dim=0)
    seq_y = torch.cat(seq_y, dim=0)

    return seq_x, seq_y




#変える部分-----------------------------------------------------------------------------------------------------------------

testloss = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\data30stride1type\3d_testloss20250523_065335.pickle"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\mixhit_fortesttype.pickle"
motor_angle = True
motor_force = True
magsensor = True
L = 30
stride = 1
testin = None
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

type_end_list = myfunction.get_type_change_end(typedf)
seq_x, seq_y = make_sequence_tensor_stride(x_change, y_data,type_end_list, L=L, stride=stride)


# モデルのロード
minid = str(get_min_loss_epoch(testloss))
print(f"使用したephoch:{minid}")
modelpath = myfunction.find_pickle_files("epoch" + minid + "_", directory=resultdir, extension='.pth')
model_from_script = torch.jit.load(modelpath, map_location="cuda:0")
model_from_script.eval()

# print(len(y_change))

dis_array = np.zeros((seq_x.shape[0]-10, 4))
esty = np.zeros((seq_x.shape[0]-10, 12))
# print(dis_array)

for i in range(seq_x.shape[0]-10):
    # sample_idx = random.randint(int(len(x_data) * 0.8 ),len(x_data)-1)  # 推論したいサンプルのインデックス
    sample_idx = i  # 推論したいサンプルのインデックス
    single_sample = seq_x[sample_idx].unsqueeze(0)  # (input_dim,) -> (1, input_dim)
    # 推論を行う（GPUが有効ならGPU上で実行）
    with torch.no_grad():  # 勾配計算を無効化
        prediction = model_from_script(single_sample)
    # print(y_change[sample_idx])
    single_sample = single_sample * x_std + x_mean
    prediction = prediction * y_std + y_mean
    distance = culc_gosa(prediction.tolist()[0], seq_y[sample_idx].tolist())
    dis_array[i, :] = distance
    esty[i,:] = prediction.tolist()[0]
end = time.time()

print(esty)

print(dis_array)
column_means = np.mean(dis_array, axis=0)
print("列ごとの平均:", column_means.round(2))


#____________________________________________________________________________________________

df_ana = pd.DataFrame(esty)
df_moto = pd.DataFrame(seq_y.cpu().numpy())

columns =  ["Mc2x","Mc2y","Mc2z","Mc3x","Mc3y","Mc3z","Mc4x","Mc4y","Mc4z","Mc5x","Mc5y","Mc5z"]
df_ana.columns = columns
df_moto.columns = columns

print(df_ana)
# 3Dプロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
origin_point = ax.scatter([0], [0], [0], color='k', s=20)
# 軸の範囲を設定
ax.set_xlim(-150, 100)
ax.set_ylim(-100, 150)
ax.set_zlim(-250, 0)
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xticklabels([])
# 初期化用のデータ（空のプロットを用意）
scatter_moto = ax.scatter([], [], [], color='r')  # 赤い点
line_moto, = ax.plot([], [], [], 'r-', label='Actual')  # 赤い線

scatter_ana = ax.scatter([], [], [], color='b')  # 青い点
line_ana, = ax.plot([], [], [], 'b-', label='estimated')  # 青い線

# アニメーションの更新関数
def update(frame):
    # フレームに応じたデータを取得
    row_moto = df_moto.iloc[frame]
    points_moto = [

        (row_moto['Mc2x'], row_moto['Mc2y'], row_moto['Mc2z']),
        (row_moto['Mc3x'], row_moto['Mc3y'], row_moto['Mc3z']),
        (row_moto['Mc4x'], row_moto['Mc4y'], row_moto['Mc4z']),
        (row_moto['Mc5x'], row_moto['Mc5y'], row_moto['Mc5z']),
    ]
    x_moto, y_moto, z_moto = zip(*points_moto)

    row_ana = df_ana.iloc[frame]
    points_ana = [

        (row_ana['Mc2x'], row_ana['Mc2y'], row_ana['Mc2z']),
        (row_ana['Mc3x'], row_ana['Mc3y'], row_ana['Mc3z']),
        (row_ana['Mc4x'], row_ana['Mc4y'], row_ana['Mc4z']),
        (row_ana['Mc5x'], row_ana['Mc5y'], row_ana['Mc5z']),
    ]
    x_ana, y_ana, z_ana = zip(*points_ana)

    # 点と線を更新
    scatter_moto._offsets3d = (x_moto, y_moto, z_moto)
    line_moto.set_data(x_moto, y_moto)
    line_moto.set_3d_properties(z_moto)

    scatter_ana._offsets3d = (x_ana, y_ana, z_ana)
    line_ana.set_data(x_ana, y_ana)
    line_ana.set_3d_properties(z_ana)

    return scatter_moto, line_moto, scatter_ana, line_ana

# アニメーションの設定 (blit=False に設定)
ani = FuncAnimation(fig, update, frames=len(df_ana), interval=100, blit=False)

plt.legend()
ax.view_init(elev=9.7, azim=-48)
ani.save('animation.mp4', writer='ffmpeg', fps=10, dpi=300, bitrate=2000)
# ① GUI表示で角度をマウス操作で調整
plt.show()

# ② 調整後の視点角度を取得
final_azim = ax.azim
final_elev = ax.elev
print(f"Final view angle: azim={final_azim}, elev={final_elev}")

# ③ 再設定（念のため）＋アニメーション保存
ax.view_init(elev=final_elev, azim=final_azim)
ani.save('animation.mp4', writer='ffmpeg', fps=10, dpi=300, bitrate=2000)
