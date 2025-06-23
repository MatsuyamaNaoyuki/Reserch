# import matplotlib.pyplot as plt
# import numpy as np
# import japanize_matplotlib

# plt.rcParams.update({'font.size': 28})

# # 値の定義（縦棒で並べる順に）
# labels = ['ResNet18', 'GRU+ResNet18', "Diffusion model"]
# x = np.arange(len(labels))  # → [0, 1]
# bar_width = 0.35

# # 値の並びをデータ中心に変更
# values = np.array([
#     [7.53, 8.84 , 7.27],
#     [0,8.68 ,6.19 ]
# ])
# name = ['連続30フレーム', '30フレーム（10フレーム間隔）']
# colors = ['#377eb8', '#ff7f00']

# # colors = ['#377eb8']
# # 描画
# fig, ax = plt.subplots(figsize=(12, 5))
# for i in range(len(values)):
#     ax.bar(x + i * bar_width, values[i], width=bar_width, label=name[i], color=colors[i], edgecolor='black')

# # 軸・ラベル・凡例
# ax.set_xticks(x + bar_width / 2)
# ax.set_xticklabels(labels)
# ax.set_ylabel('Estimation Error [mm]')
# ax.set_ylim(0, 12)
# # ax.set_title('磁気センサー有無による推定誤差')
# ax.legend()
# ax.grid(axis='y', linestyle='-', alpha=0.7)

# plt.tight_layout()
# plt.show()

#=======================================================================
# import matplotlib.pyplot as plt
# import numpy as np
# import japanize_matplotlib

# plt.rcParams.update({'font.size': 28})

# labels = ['ResNet18', 'GRU+ResNet18', "Diffusion model"]
# x = np.arange(len(labels))
# bar_width = 0.35

# values = np.array([
#     [7.53, 8.84, 7.27],
#     [0.00, 8.68, 6.19]
# ])
# name = ['連続30フレーム', '30フレーム（10フレーム間隔）']
# colors = ['#377eb8', '#ff7f00']

# fig, ax = plt.subplots(figsize=(12, 5))

# # --- 特別に左の1本だけ赤くして別ラベルに ---
# ax.bar(x[0], values[0][0], width=bar_width, color='#03AF7A', edgecolor='black', label='1フレーム')

# # --- 残りの棒グラフ ---
# for i in range(len(values)):
#     for j in range(len(x)):
#         if not (i == 0 and j == 0):  # 左上の赤バーはもう描いたのでスキップ
#             ax.bar(x[j] + i * bar_width, values[i][j], width=bar_width, color=colors[i], edgecolor='black', label=name[i] if j == 0 else "")

# # 軸・ラベル・凡例
# ax.set_xticks(x + bar_width / 2)
# ax.set_xticklabels(labels)
# ax.set_ylabel('Estimation Error [mm]')
# ax.set_ylim(0, 12)
# ax.legend()
# ax.grid(axis='y', linestyle='-', alpha=0.7)

# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

plt.rcParams.update({'font.size': 24})

labels = ['ResNet18', 'GRU+ResNet18', 'Diffusion model']
x = np.arange(len(labels))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))

# グラフの描画
# ResNet18：中央に1本（緑）
ax.bar(x[0] + bar_width/2, 7.53, width=bar_width, color='#ff7f00', edgecolor='black', label='1フレーム')

# GRU+ResNet18：青とオレンジを左右に
ax.bar(x[1] - bar_width/2, 6.77, width=bar_width, color='#377eb8', edgecolor='black', label='32フレーム（連続）')
ax.bar(x[1] + bar_width/2, 6.28, width=bar_width, color= '#4daf4a', edgecolor='black', label='32フレーム（10フレーム間隔）')

# Diffusion model：青とオレンジを左右に
ax.bar(x[2] - bar_width/2, 7.27, width=bar_width, color='#377eb8', edgecolor='black')
ax.bar(x[2] + bar_width/2, 6.19, width=bar_width, color= '#4daf4a', edgecolor='black')

# 軸と凡例
xticks_pos = [x[0] + bar_width / 2, x[1], x[2]]       
ax.set_xticks(xticks_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Estimation Error [mm]')
ax.set_ylim(0, 11)
ax.legend()
ax.grid(axis='y', linestyle='-', alpha=0.7)

plt.tight_layout()
plt.show()
