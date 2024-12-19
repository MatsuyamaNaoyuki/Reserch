import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# データの準備
csv_file_path = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\dataset_margemag_tewnty.csv"
df_moto = pd.read_csv(csv_file_path)

csv_file_path = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\output.csv"
df_ana = pd.read_csv(csv_file_path)

df_moto = df_moto.drop(['time', 'rotate1', 'rotate2', 'rotate3', 'rotate4', 'force1', 'force2', 'force3', 'force4'], axis=1)
df_moto = df_moto.drop(['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9'], axis=1)


# 3Dプロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 軸の範囲を設定
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.set_zlim(-200, 0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 初期化用のデータ（空のプロットを用意）
scatter_moto, = ax.plot([], [], [], 'ro', color = 'r', label='推定前', markersize=8)  # 点
line_moto, = ax.plot([], [], [], 'r-', color = 'r')  # 線


scatter_ana, = ax.plot([], [], [], 'ro', color = 'b', label='推定後', markersize=8)
line_ana, = ax.plot([], [], [], 'r-',  color = 'b')  # 1本目の線

# アニメーションの更新関数
def update(frame):
    # フレームに応じたデータを取得
    row = df_moto.iloc[frame]
    points_moto = [
        (0, 0, 0),
        (row['Mc2x'], row['Mc2y'], row['Mc2z']),
        (row['Mc3x'], row['Mc3y'], row['Mc3z']),
        (row['Mc4x'], row['Mc4y'], row['Mc4z']),
        (row['Mc5x'], row['Mc5y'], row['Mc5z']),
    ]
    x_moto, y_moto, z_moto = zip(*points_moto)

    row = df_ana.iloc[frame]
    points_ana = [
        (0, 0, 0),
        (row['Mc2x'], row['Mc2y'], row['Mc2z']),
        (row['Mc3x'], row['Mc3y'], row['Mc3z']),
        (row['Mc4x'], row['Mc4y'], row['Mc4z']),
        (row['Mc5x'], row['Mc5y'], row['Mc5z']),
    ]
    x_ana, y_ana, z_ana = zip(*points_ana)

    # 点と線を更新
    scatter_moto.set_data(x_moto, y_moto)
    scatter_moto.set_3d_properties(z_moto)
    line_moto.set_data(x_moto[1:], y_moto[1:])
    line_moto.set_3d_properties(z_moto[1:])

    scatter_ana.set_data(x_ana, y_ana)
    scatter_ana.set_3d_properties(z_ana)
    line_ana.set_data(x_ana[1:], y_ana[1:])
    line_ana.set_3d_properties(z_ana[1:])

    return scatter_moto, line_ana, scatter_ana, line_moto

# アニメーションの設定
ani = FuncAnimation(fig, update, frames=len(range(0, 59)), interval=100, blit=False)

plt.legend()
plt.show()
