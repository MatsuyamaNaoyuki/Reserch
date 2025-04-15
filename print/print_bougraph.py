import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 22})
# データの準備（4カテゴリ × 4グループ）
categories = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
values = np.array([
    [0.54, 1.34, 2.17, 4.56],  # グループ1
    [0.91, 2.29, 3.95, 6.63],  # グループ2
    [0.68, 1.69, 2.79, 4.89 ],   # グループ3
    [0.55, 1.36, 2.57, 5.80]   # グループ4
])

# values = np.array([
#     [1.16, 2.37, 4.17, 6.48],  # グループ1
#     [1.49, 3.42, 6.21, 8.74],  # グループ2
#     [1.12, 2.23, 3.87, 5.66 ],   # グループ3
#     [1.10, 2.14, 3.75, 6.43]   # グループ4
# ])

# values = np.array([
#     [0.96, 2.30, 3.51, 5.67],  # グループ1
#     [1.52, 3.75, 5.88, 8.76],  # グループ2
#     [1.25, 3.02, 4.71, 7.56 ],   # グループ3
#     [1.02, 2.49, 4.10, 7.27]   # グループ4
# ])

# values = np.array([
#     [1.49, 3.34, 5.50, 7.53],  # グループ1
#     [2.40, 6.22, 10.37, 14.17],  # グループ2
#     [1.68, 3.84, 6.13, 8.68 ],   # グループ3
#     [1.53, 3.27, 5.35, 7.89]   # グループ4
# ])



name = ["ModelA", "ModelB", "ModelC", "ModelD"]
# パラメータ設定
num_groups = values.shape[0]  # データセットの数（4つ）
num_categories = values.shape[1]  # カテゴリの数（4つ）
bar_width = 0.2  # 棒の幅

# x軸の位置を設定
x = np.arange(num_categories)

# 色の設定（各データセットの色を変える）
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#984ea3']

# グラフの描画
plt.figure(figsize=(8, 5))
for i in range(num_groups):
    plt.bar(x + (i - num_groups/2) * bar_width + bar_width/2, values[i], width=bar_width, label=name[i], color=colors[i], edgecolor='black')

# x軸のラベルを設定
plt.ylim(0,15)
plt.xticks(x, categories)

# ラベルとタイトル

plt.ylabel('Estimation Error [mm]')
# plt.title('Grouped Bar Chart (4×4)')

# 凡例を追加
plt.legend()

# グリッドを追加
plt.grid(axis='y', linestyle='--', alpha=0.7)

# グラフを表示
plt.show()
