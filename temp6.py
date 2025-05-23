import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
 



def update(frames):
    data.append(2)
    x = range(frames)
    y = data[:frames]
    print(f"x:{x}")
    print(f"y:{y}")
    ax.plot(x, y)

    
    data2.append(3)
    x1 = range(frames)
    y1 = data2[:frames]
    
    ax.plot(x1,y1)
    
 
data = []
data2 = []
 
# 2.グラフ領域の作成
fig, ax = plt.subplots()
 
# 3. グラフの初期設定
my_line, = ax.plot([], [])



r = 10
ax.set_xlim(0, 20)
ax.set_ylim(0, 6)
ax.set_aspect('equal') # グラフのアスペクト比を１：１に設定
 

    
# 5. アニメーション化
anim = FuncAnimation(fig, update, frames=20)
plt.show()



# def mag_data_change(row):
#     split_value = row.split('/')
#     if len(split_value) != 9:
#         split_value = split_value[1:]
#     int_list = [int(c) for c in split_value] 
#     return int_list

# def update(frame):
#     magdata = Ms.get_value()
#     row = mag_data_change(magdata)
#     # df.loc[len(df)] = data[frame]  # データ追加

#     # for i, col in enumerate(columns):
#     #     lines[i].set_data(range(len(df)), df[col])  # 線データ更新
#     line.set_data(range(len(df)), row[0])
#     return lines






# Ms = MagneticSensor()

# columns = ['sensor1', 'sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9']
# sensordf = pd.DataFrame(columns=columns)
# for i in range(10):
#     magdata = Ms.get_value()
#     row = mag_data_change(magdata)
#     sensordf.loc[len(sensordf)] = row
    
    
    
    
    
# print(sensordf)


# fig, ax = plt.subplots()
# # lines = [ax.plot([], [], label=col)[0] for col in columns]
# line = ax.plot([],[])
# ax.set_xlim(0, 1000)
# ax.set_ylim(500, 1000)
# ax.set_title("追加されていくデータの可視化")
# ax.legend()

# ani = animation.FuncAnimation(fig, update, frames=1000, interval=1000, blit=True)
# plt.show()