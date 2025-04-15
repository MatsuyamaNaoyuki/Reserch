import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

plt.rcParams.update({'font.size': 18})

# 値の定義（縦棒で並べる順に）
labels = ['磁気センサーあり', '磁気センサーなし']
x = np.arange(len(labels))  # → [0, 1]
bar_width = 0.25

# 値の並びをデータ中心に変更
values = np.array([
    [6.48, 8.74],  # 接触ありのみ
    [7.53, 14.17], # 接触あり＆接触なし
])
name = ['接触ありのみ', '接触あり＆接触なし']
colors = ['#377eb8', '#ff7f00']

# 描画
fig, ax = plt.subplots(figsize=(6, 5))
for i in range(len(values)):
    ax.bar(x + i * bar_width, values[i], width=bar_width, label=name[i], color=colors[i], edgecolor='black')

# 軸・ラベル・凡例
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(labels)
ax.set_ylabel('Estimation Error [mm]')
ax.set_ylim(0, 15)
ax.set_title('磁気センサー有無による推定誤差')
ax.legend()
ax.grid(axis='y', linestyle='-', alpha=0.7)

plt.tight_layout()
plt.show()
