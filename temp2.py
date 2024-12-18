import csv, pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from myclass import myfunction
# import japanize_matplotlib


file_path = "C:\\Users\\shigf\\Program\\data\\margedata_formotioncapture20241217_230409.pickle"
dataset = pd.read_pickle(file_path)

# データ型を確認して適切に変換
if isinstance(dataset, list):
    dataset = pd.DataFrame(dataset)

# 必要な範囲（行1～2000、27列目）を取得
dataset = dataset.iloc[1:2001, [27]]

# 列名を設定
dataset.columns = ['Mc5z']

# 大きな値を削除（100000未満のみ）
dataset = dataset[dataset['Mc5z'] < 100000]


# print(dataset.mean())
# print(dataset.std())


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dataset.index, dataset.values, label="MotionCapture Value")

# グラフの設定
ax.set_title("After deletion of outliers", fontsize=20)
ax.set_xlabel("time inex", fontsize=16)
ax.set_ylabel("Values", fontsize=16)
ax.legend(fontsize=12)

# 軸のフォントサイズ設定
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.show()
