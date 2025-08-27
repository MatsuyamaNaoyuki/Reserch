import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from myclass import myfunction
import numpy as np

y_data = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\Reserch\real20250821_135157.pickle")
estimation_array= myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\Reserch\result20250821_133614.pickle")


# visual_index = range(0,len(y_data),10)
list_estimation_array = [t.cpu().tolist()[0] for t in estimation_array]

# print(list_ydata)
# estimation_array = estimation_array[::5]
# list_ydata = list_ydata[::5]
# list_ydata = [list_ydata[i] for i in visual_index]


half= len(y_data) // 2

x  = [row[9]  for row in y_data[:half:5]]
y  = [row[10] for row in y_data[:half:5]]
z  = [row[11] for row in y_data[:half:5]]

ex = [row[9]  for row in list_estimation_array[:half:5]]
ey = [row[10] for row in list_estimation_array[:half:5]]
ez = [row[11] for row in list_estimation_array[:half:5]]


vx = [ai - bi for ai, bi in zip(ex, x)]
vy = [ai - bi for ai, bi in zip(ey, y)]
vz = [ai - bi for ai, bi in zip(ez, z)]

magnitude = np.sqrt(np.array(vx)**2 + np.array(vy)**2 + np.array(vz)**2)

# カラーマップを作成
norm = plt.Normalize(magnitude.min(), magnitude.max())
colors = cm.viridis(norm(magnitude))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 点群をプロット
ax.scatter(x, y, z, marker='.', color=colors)
ax.scatter(ex, ey, ez, marker='.', color=colors)
ax.quiver(x, y, z, vx, vy, vz, length=1, normalize=False, color=colors,arrow_length_ratio=0.5)
# ラベル付け
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

max_range = max(
    max(x) - min(x),
    max(y) - min(y),
    max(z) - min(z),
    max(ex) - min(ex),
    max(ey) - min(ey),
    max(ez) - min(ez)
) / 2.0

mid_x = (max(x) + min(x)) / 2.0
mid_y = (max(y) + min(y)) / 2.0
mid_z = (max(z) + min(z)) / 2.0

ax.set_box_aspect([1, 1, 1])  # 各軸の比率を同じに設定
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
# 表示
plt.show()