import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

pd.options.display.float_format = '{:.2f}'.format

csv_file_path = r"C:\Users\shigf\Downloads\springmcdata.pickle"
df = pd.read_pickle(csv_file_path)
df = df[0]

# 点の作成
points = [
    (0, 0, 0),
    (df[4] - df[1], df[5] - df[2], df[6] - df[3]),
    (df[7] - df[1], df[8] - df[2], df[9] - df[3]),
    (df[10] - df[1], df[11] - df[2], df[12] - df[3]),
    (df[13] - df[1], df[14] - df[2], df[15] - df[3]),
]

x, y, z = zip(*points)

# プロットの準備
fig = plt.figure()
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
ax = fig.add_subplot(111, projection='3d')

# 点と線を描画
for i in range(len(x)):
    ax.scatter(x[i], y[i], z[i], color='r', s=30)
ax.plot(x[1:], y[1:], z[1:], color='r')

# -----------------------
# 軸スケールの統一処理
# -----------------------
# 全軸の最大範囲を使って等スケールに
x_range = np.ptp(x)  # peak-to-peak: max - min
y_range = np.ptp(y)
z_range = np.ptp(z)
max_range = max(x_range, y_range, z_range)

# 中心を計算
x_middle = (max(x) + min(x)) / 2
y_middle = (max(y) + min(y)) / 2
z_middle = (max(z) + min(z)) / 2

# 軸範囲を統一してセット
ax.set_xlim(x_middle - max_range / 2, x_middle + max_range / 2)
ax.set_ylim(y_middle - max_range / 2, y_middle + max_range / 2)
ax.set_zlim(z_middle - max_range / 2, z_middle + max_range / 2)

# 目盛もきれいに（例: 20単位）
def nice_ticks(vmin, vmax, step=20):
    start = int(np.floor(vmin / step) * step)
    end = int(np.ceil(vmax / step) * step)
    return np.arange(start, end + step, step)

ax.set_xticks(nice_ticks(*ax.get_xlim(), step=30))
ax.set_yticks(nice_ticks(*ax.get_ylim(), step=30))
ax.set_zticks(nice_ticks(*ax.get_zlim(), step=30))

# 等スケール表示
ax.set_box_aspect([1, 1, 1])  # X:Y:Z = 1:1:1

# 凡例を消す
ax.legend().remove()

plt.show()
