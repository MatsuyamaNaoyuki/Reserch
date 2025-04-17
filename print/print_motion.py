import pickle 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# CSVファイルの読み込み
file_path = r"C:\Users\WRS\Desktop\Matsuyama\Reserch\tube_softfinger_test_10_20250415_172035.pickle"
data = pd.read_pickle(file_path)
selected_columns = data[['Mc5x', 'Mc5y', 'Mc5z']]
print(selected_columns)
print(selected_columns.dtypes)



grid_size = 1.0  # 1.0の精度で丸める
rounded_data = selected_columns.applymap(lambda x: round(x / grid_size) * grid_size)

# ユニークなデータを抽出
unique_data = rounded_data.drop_duplicates()

print(len(rounded_data))
print(len(unique_data))

x = unique_data['Mc5x'].tolist()
y = unique_data['Mc5y'].tolist()
z = unique_data['Mc5z'].tolist()



x.append(0)
y.append(0)
z.append(0)


# 3Dプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 点群をプロット
ax.scatter(x, y, z, c='r', marker='o')

# ラベル付け
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

max_range = max(
    max(x) - min(x),
    max(y) - min(y),
    max(z) - min(z)
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