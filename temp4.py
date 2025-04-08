import matplotlib.pyplot as plt
import numpy as np

# データの準備（4カテゴリ × 4グループ）
categories = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
values = np.array([
    [0.54, 1.34, 2.17, 4.56],  # グループ1
    [0.91, 2.29, 3.95, 6.63],  # グループ2
    [1, 1, 1,1 ],   # グループ3
    [0.55, 1.36, 2.57, 5.80]   # グループ4
])


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
plt.xticks(x, categories)

# ラベルとタイトル

plt.ylabel('Estimation Error (mm)')
plt.title('Grouped Bar Chart (4×4)')

# 凡例を追加
plt.legend()

# グリッドを追加
plt.grid(axis='y', linestyle='--', alpha=0.7)

# グラフを表示
plt.show()
