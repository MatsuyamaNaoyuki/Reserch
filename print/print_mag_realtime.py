import sys
sys.path.append('.')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from myclass.MyMagneticSensor import MagneticSensor
import pandas as pd

frame_num =200


def mag_data_change(row):
    split_value = row.split('/')
    if len(split_value) != 9:
        split_value = split_value[1:]
    int_list = [int(c) for c in split_value] 
    return int_list


def update(frame):
    global sensordf          # ← リセットのために global 宣言

    # ── フレーム 0 になったら軸とデータをクリア ──
    if frame == 0:
        sensordf = pd.DataFrame(columns=columns)   # DataFrame を空に
        for ax in (ax1, ax2, ax3):
            ax.cla()                               # 軸をまっさらに
            ax.set_xlim(0, frame_num)
            ax.set_ylim(300, 900)

    # ---------- ここから毎フレームの処理 ----------
    magdata = Ms.get_value()
    sensordf.loc[len(sensordf)] = mag_data_change(magdata)

    x = range(len(sensordf))        # ← DataFrame の長さに合わせる
    ax1.plot(x, sensordf["sensor1"], color='blue')
    ax1.plot(x, sensordf["sensor2"], color='red')
    ax1.plot(x, sensordf["sensor3"], color='green')

    ax2.plot(x, sensordf["sensor4"], color='blue')
    ax2.plot(x, sensordf["sensor5"], color='red')
    ax2.plot(x, sensordf["sensor6"], color='green')

    ax3.plot(x, sensordf["sensor7"], color='blue')
    ax3.plot(x, sensordf["sensor8"], color='red')
    ax3.plot(x, sensordf["sensor9"], color='green')

    
    
Ms = MagneticSensor()
columns = ['sensor1', 'sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9']
# columns = ['sensor1', 'sensor2','sensor3']
sensordf = pd.DataFrame(columns=columns)


# data = []
# data2 = []
 
# 2.グラフ領域の作成

fig = plt.figure()

ax1 = plt.subplot(221) 
ax2 = plt.subplot(222) 
ax3 = plt.subplot(223) 
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, frame_num)
    ax.set_ylim(550, 650)
    
# 5. アニメーション化
anim = FuncAnimation(fig, update, frames=frame_num, interval=10, repeat=True)
plt.show()



# import sys
# sys.path.append('.')
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
# from myclass.MyMagneticSensor import MagneticSensor
# import pandas as pd

# frame_num =200


# def mag_data_change(row):
#     split_value = row.split('/')
#     if len(split_value) != 9:
#         split_value = split_value[1:]
#     int_list = [int(c) for c in split_value] 
#     return int_list


# def update(frame):
#     global sensordf          # ← リセットのために global 宣言

#     # ── フレーム 0 になったら軸とデータをクリア ──
#     if frame == 0:
#         sensordf = pd.DataFrame(columns=columns)   # DataFrame を空に
#         for ax in (ax1):
#             ax.cla()                               # 軸をまっさらに
#             ax.set_xlim(0, frame_num)
#             ax.set_ylim(500, 900)

#     # ---------- ここから毎フレームの処理 ----------
#     magdata = Ms.get_value()
#     sensordf.loc[len(sensordf)] = mag_data_change(magdata)

#     x = range(len(sensordf))        # ← DataFrame の長さに合わせる
#     ax1.plot(x, sensordf["sensor1"], color='blue')
#     ax1.plot(x, sensordf["sensor2"], color='red')
#     ax1.plot(x, sensordf["sensor3"], color='green')

   

    
    
# Ms = MagneticSensor()
# # columns = ['sensor1', 'sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9']
# columns = ['sensor1', 'sensor2','sensor3']
# sensordf = pd.DataFrame(columns=columns)


# # data = []
# # data2 = []
 
# # 2.グラフ領域の作成

# fig = plt.figure()

# ax1 = plt.subplot(221) 


# ax1.set_xlim(0, frame_num)
# ax1.set_ylim(550, 650)
    
# # 5. アニメーション化
# anim = FuncAnimation(fig, update, frames=frame_num, interval=10, repeat=True)
# plt.show()





