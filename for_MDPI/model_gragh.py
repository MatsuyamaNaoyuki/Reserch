
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
ax.bar(x[0] + bar_width/2, 4.9575, width=bar_width, color='#ff7f00', edgecolor='black', label='only 1 frame')

# GRU+ResNet18：青とオレンジを左右に
ax.bar(x[1] - bar_width/2, 3.8275, width=bar_width, color='#377eb8', edgecolor='black', label='32 consecutive frames')
ax.bar(x[1] + bar_width/2, 3.1775, width=bar_width, color= '#4daf4a', edgecolor='black', label='32 frames with 10-frame intervals')

# Diffusion model：青とオレンジを左右に
ax.bar(x[2] - bar_width/2, 4.0925, width=bar_width, color='#377eb8', edgecolor='black')
ax.bar(x[2] + bar_width/2, 3.465, width=bar_width, color= '#4daf4a', edgecolor='black')

# 軸と凡例
xticks_pos = [x[0] + bar_width / 2, x[1], x[2]]       
ax.set_xticks(xticks_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Estimation Error [mm]')
ax.set_ylim(0, 6)
ax.legend()
ax.grid(axis='y', linestyle='-', alpha=0.7)

plt.tight_layout()
plt.show()
