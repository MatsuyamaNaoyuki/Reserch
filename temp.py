import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

plt.rcParams.update({'font.size': 22})
# データの準備（4カテゴリ × 4グループ）
categories = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
values = np.array([
    [ 2.49 , 6.32, 11.63, 19.28],  # グループ1
    [3.08,  7.81 ,14.66 ,23.48],  # グループ2
    [2.5   ,7.25, 15.76, 27.55],   # グループ3
    [2.84 , 7.28, 15.33 ,26.3]   # グループ4
])



name = ["連続時間変更なし", "連続時間変更有", "10飛ばし時間変更なし", "10飛ばし時間変更有"]
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
plt.ylim(0,30)
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
