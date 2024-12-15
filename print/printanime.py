import pandas as pd
import matplotlib.animation as anm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import time

# CSVファイルのパスを指定
csv_file_path = 'MagChack20241014_234007.csv'
df = pd.read_csv(csv_file_path, header=None)



first_time = df[0][0]
first_time = datetime.strptime(first_time, "%Y-%m-%d %H:%M:%S.%f")
for i in range(len(df)):
    tmptime = datetime.strptime(df[0][i],  "%Y-%m-%d %H:%M:%S.%f")
    df[1][i] = df[1][i] * 5 / 1024.0
    seconds = tmptime - first_time
    milliseconds = seconds.total_seconds() * 1000
    df[0][i] = milliseconds

print(df)

x_values = df[0]  # 横軸はデータ数（インデックス）
y_values_1 = df[1]  # 1列目

fig = plt.figure(figsize=(10, 6))
plt.xlim(0,1000)

def update(i, x_values, y_values):
    if i != 0:
        plt.cla()                      # 現在描写されているグラフを消去
    X = x_values[0:i]
    Y = y_values[0:i]
    plt.xlim(0,43000)
    plt.ylim(0,5)
    plt.xlabel('time [ms]', size = 18)
    plt.ylabel('sensor Value [V]', size = 18)
    plt.grid()
    plt.plot(X, Y)


ani = anm.FuncAnimation(fig, update, fargs = (x_values, y_values_1), \
    interval = 100, frames = len(x_values))

ani.save("Sample.gif", writer = 'imagemagick')