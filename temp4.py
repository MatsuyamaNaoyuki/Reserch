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


def update(frames):
    magdata = Ms.get_value()
    row = mag_data_change(magdata)
    sensordf.loc[len(sensordf)] = row
  
    x = range(frames+1)
    y1 = sensordf.loc[:frames, "sensor1"].tolist()
    y2 = sensordf.loc[:frames, "sensor2"].tolist()
    y3 = sensordf.loc[:frames, "sensor3"].tolist()

    ax1.plot(x,y1,color='blue')
    ax1.plot(x,y2,color='red')
    ax1.plot(x,y3,color='green')
    
    y4 = sensordf.loc[:frames, "sensor4"].tolist()
    y5 = sensordf.loc[:frames, "sensor5"].tolist()
    y6 = sensordf.loc[:frames, "sensor6"].tolist()

    ax2.plot(x,y4,color='blue')
    ax2.plot(x,y5,color='red')
    ax2.plot(x,y6,color='green')
    
    y7 = sensordf.loc[:frames, "sensor7"].tolist()
    y8 = sensordf.loc[:frames, "sensor8"].tolist()
    y9 = sensordf.loc[:frames, "sensor9"].tolist()

    ax3.plot(x,y7,color='blue')
    ax3.plot(x,y8,color='red')
    ax3.plot(x,y9,color='green')
    
    
Ms = MagneticSensor()
columns = ['sensor1', 'sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9']
sensordf = pd.DataFrame(columns=columns)


# data = []
# data2 = []
 
# 2.グラフ領域の作成

fig = plt.figure()

ax1 = plt.subplot(221) 
ax2 = plt.subplot(222) 
ax3 = plt.subplot(223) 
ax1.set_xlim(0, frame_num)
ax1.set_ylim(500, 800)
ax2.set_xlim(0, frame_num)
ax2.set_ylim(500, 800)
ax3.set_xlim(0, frame_num)
ax3.set_ylim(500, 800)

    
# 5. アニメーション化
anim = FuncAnimation(fig, update, frames=frame_num, interval=10)
plt.show()




