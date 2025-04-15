import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# データの準備
front = 388
back = 518
csv_file_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_withhit\withhit_fortest20250227_134803.pickle"
df_moto = pd.read_pickle(csv_file_path)
df_moto = df_moto.iloc[front:back]

csv_file_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\panademo\output2.pickle"
df_ana = pd.read_pickle(csv_file_path)

df_moto = df_moto.drop(['time', 'rotate1', 'rotate2', 'rotate3', 'rotate4', 'force1', 'force2', 'force3', 'force4'], axis=1)
df_moto = df_moto.drop(['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9'], axis=1)

print(df_ana)
# 3Dプロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 軸の範囲を設定
ax.set_xlim(-150, 100)
ax.set_ylim(-100, 150)
ax.set_zlim(-250, 0)
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xticklabels([])
# 初期化用のデータ（空のプロットを用意）
scatter_moto = ax.scatter([], [], [], color='r')  # 赤い点
line_moto, = ax.plot([], [], [], 'r-', label='Actual')  # 赤い線

scatter_ana = ax.scatter([], [], [], color='b')  # 青い点
line_ana, = ax.plot([], [], [], 'b-', label='estimated')  # 青い線

# アニメーションの更新関数
def update(frame):
    # フレームに応じたデータを取得
    row_moto = df_moto.iloc[frame]
    points_moto = [

        (row_moto['Mc2x'], row_moto['Mc2y'], row_moto['Mc2z']),
        (row_moto['Mc3x'], row_moto['Mc3y'], row_moto['Mc3z']),
        (row_moto['Mc4x'], row_moto['Mc4y'], row_moto['Mc4z']),
        (row_moto['Mc5x'], row_moto['Mc5y'], row_moto['Mc5z']),
    ]
    x_moto, y_moto, z_moto = zip(*points_moto)

    row_ana = df_ana.iloc[frame]
    points_ana = [

        (row_ana['Mc2x'], row_ana['Mc2y'], row_ana['Mc2z']),
        (row_ana['Mc3x'], row_ana['Mc3y'], row_ana['Mc3z']),
        (row_ana['Mc4x'], row_ana['Mc4y'], row_ana['Mc4z']),
        (row_ana['Mc5x'], row_ana['Mc5y'], row_ana['Mc5z']),
    ]
    x_ana, y_ana, z_ana = zip(*points_ana)

    # 点と線を更新
    scatter_moto._offsets3d = (x_moto, y_moto, z_moto)
    line_moto.set_data(x_moto, y_moto)
    line_moto.set_3d_properties(z_moto)

    scatter_ana._offsets3d = (x_ana, y_ana, z_ana)
    line_ana.set_data(x_ana, y_ana)
    line_ana.set_3d_properties(z_ana)

    return scatter_moto, line_moto, scatter_ana, line_ana

# アニメーションの設定 (blit=False に設定)
ani = FuncAnimation(fig, update, frames=len(df_ana), interval=100, blit=False)

plt.legend()
plt.show()
ax.view_init(elev=0, azim=0)
ani.save('animation.mp4', writer='ffmpeg', fps=10, dpi=300, bitrate=2000)
