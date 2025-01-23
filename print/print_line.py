import pickle 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# CSVファイルの読み込み
file_path = r"C:\Users\shigf\Program\data\sentan_newcam\modifydata20250122.csv"
data = pd.read_csv(file_path)

selected_columns = data[['Mc2x', 'Mc2y', 'Mc2z','Mc3x', 'Mc3y', 'Mc3z','Mc4x', 'Mc4y', 'Mc4z','Mc5x', 'Mc5y', 'Mc5z']]
print(selected_columns)
print(selected_columns.dtypes)



grid_size = 10.0  # 1.0の精度で丸める
rounded_data = selected_columns.applymap(lambda x: round(x / grid_size) * grid_size)

# ユニークなデータを抽出
unique_data = rounded_data.drop_duplicates()

print(len(rounded_data))
print(len(unique_data))
x2 = unique_data['Mc2x'].tolist()
y2 = unique_data['Mc2y'].tolist()
z2 = unique_data['Mc2z'].tolist()

x3 = unique_data['Mc3x'].tolist()
y3 = unique_data['Mc3y'].tolist()
z3 = unique_data['Mc3z'].tolist()

x4 = unique_data['Mc4x'].tolist()
y4 = unique_data['Mc4y'].tolist()
z4 = unique_data['Mc4z'].tolist()

x5 = unique_data['Mc5x'].tolist()
y5 = unique_data['Mc5y'].tolist()
z5 = unique_data['Mc5z'].tolist()


points_list = [
    (x2, y2, z2),
    (x3, y3, z3),
    (x4, y4, z4),
    (x5, y5, z5),
]


# 3Dプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 点群をプロット
ax.scatter(x2, y2, z2, c='r', marker='o', s=1)
ax.scatter(x3, y3, z3, c='r', marker='o', s=1)
ax.scatter(x4, y4, z4, c='r', marker='o', s=1)
ax.scatter(x5, y5, z5, c='r', marker='o', s=1)
for i in range(100):  # 各行のインデックス
    ax.plot(
        [x2[i], x3[i], x4[i], x5[i]],  # x座標をつなぐ
        [y2[i], y3[i], y4[i], y5[i]],  # y座標をつなぐ
        [z2[i], z3[i], z4[i], z5[i]],  # z座標をつなぐ
        c='b'  # 線の色
    )


# ラベル付け
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

max_range = max(
    max(x5) - min(x5),
    max(y5) - min(y5),
    max(z5) - min(z5)
) / 2.0

mid_x = (max(x5) + min(x5)) / 2.0
mid_y = (max(y5) + min(y5)) / 2.0
mid_z = (max(z5) + min(z5)) / 2.0

ax.set_box_aspect([1, 1, 1])  # 各軸の比率を同じに設定
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
# 表示
plt.show()