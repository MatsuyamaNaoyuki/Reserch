import pickle 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


with open("1204_100data_maybeOK\\motioncapture20241205_005155.pickle", mode='br') as fi:
  motiondata = pickle.load(fi)
motiondata = [row[1:] for row in motiondata]
motiondata = [row for row in motiondata if row[12] <= 10000]
motiondata = [row for row in motiondata if row[13] <= 10000]
motiondata = [row for row in motiondata if row[14] <= 10000]


grid_size = 1.0  # 1.0の精度で丸める
rounded_data = [[round(x / grid_size) * grid_size for x in row] for row in motiondata]
unique_data = [list(item) for item in set(tuple(row) for row in rounded_data)]
print(len(rounded_data))
print(len(unique_data))

x = []
y = []
z = []



for i in range(len(unique_data)):
  x.append(unique_data[i][12])
  y.append(unique_data[i][13])
  z.append(unique_data[i][14])







# 3Dプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 点群をプロット
ax.scatter(x, y, z, c='r', marker='o')

# ラベル付け
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 表示
plt.show()