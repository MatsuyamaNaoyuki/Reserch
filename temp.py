
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

plt.rcParams.update({'font.size': 22})
# データの準備（4カテゴリ × 4グループ）
# categories = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
categories = ["磁気センサーあり\n(モーター回転角と電流値あり)", "磁気センサーなし\n(モーター回転角と電流値あり)"]
values = np.array([ 
    [6.48, 8.74],  # グループ1
    [7.53, 14.17],   # グループ3
  # グループ4
])
7.53
['接触ありのみ', '接触あり＆接触なし']

name = ['接触ありのみ', '接触あり＆接触なし']
# パラメータ設定
num_groups = values.shape[0]  # データセットの数（4つ）
num_categories = values.shape[1]  # カテゴリの数（4つ）
bar_width = 0.2  # 棒の幅

# x軸の位置を設定
x = np.array([0, 0.6]) 

# 色の設定（各データセットの色を変える）
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#984ea3']

# グラフの描画
plt.figure(figsize=(4.5, 5))  # 横6、縦5
for i in range(num_groups):
    plt.bar(x + (i - num_groups/2) * bar_width + bar_width/2, values[i], width=bar_width, label=name[i], color=colors[i], edgecolor='black')

# x軸のラベルを設定
plt.ylim(0,15)
plt.xticks(x, categories)
plt.title('磁気センサー有無による推定誤差の比較')
# ラベルとタイトル

plt.ylabel('Estimation Error [mm]')
# plt.title('Grouped Bar Chart (4×4)')

# 凡例を追加
plt.legend()

# グリッドを追加
plt.grid(axis='y', linestyle='--', alpha=0.7)

# グラフを表示
plt.show()  