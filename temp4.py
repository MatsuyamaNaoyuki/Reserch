import csv, pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from myclass import myfunction
# import japanize_matplotlib



file_path = "C:\\Users\\shigf\\Program\\DXhub\\motioncapture20241205_005155.pickle"
dataset = pd.read_pickle(file_path)
dataset = pd.DataFrame(dataset)

print(dataset.columns)
dataset = dataset.loc[:2000, 10]

# print(dataset.mean())
# print(dataset.std())


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dataset.index, dataset.values, label="MotionCapture Value")

# グラフの設定
ax.set_title("Before deletion of outliers", fontsize=20)
ax.set_xlabel("time inex", fontsize=16)
ax.set_ylabel("Values", fontsize=16)
ax.legend(fontsize=12)

# 軸のフォントサイズ設定
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.show()
