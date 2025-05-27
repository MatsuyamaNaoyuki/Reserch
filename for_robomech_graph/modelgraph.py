import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

plt.rcParams.update({'font.size': 28})

# 値の定義（縦棒で並べる順に）
labels = ['ResNet18', 'GRU+ResNet18', "Diffusion model"]
x = np.arange(len(labels))  # → [0, 1]
bar_width = 0.35

# 値の並びをデータ中心に変更
values = np.array([
    [7.53, 6.77 , 7.27]
    # [0,6.28 ,6.19 ]
])
# name = ['接触ありのみ', '接触あり＆接触なし']
# colors = ['#377eb8', '#ff7f00']

colors = ['#377eb8']
# 描画
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(len(values)):
    ax.bar(x , values[i], width=bar_width, color=colors[i], edgecolor='black')

# 軸・ラベル・凡例
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Estimation Error [mm]')
ax.set_ylim(0, 10)
# ax.set_title('磁気センサー有無による推定誤差')
# ax.legend()
ax.grid(axis='y', linestyle='-', alpha=0.7)

plt.tight_layout()
plt.show()
