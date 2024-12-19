import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def calculate_normal_vector(vec1, vec2):
    normal_vector = np.cross(vec1, vec2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector

def project_points_onto_plane(df, point1, point2):

    vec1 = np.array([point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]])
    vec2 = np.array([0, 0,  1])
    norm_vec = calculate_normal_vector(vec1, vec2)

    # 投影を計算する関数
    def project_single_point(x, y, z):
        targetvec = np.array([x,y,z])
        B = np.dot(norm_vec, targetvec) * norm_vec
        projected_point = targetvec- B
        return projected_point[0], projected_point[1]

    # 各列を投影
    projected_data = {}
    for prefix in ['Mc2', 'Mc3', 'Mc4', 'Mc5']:
        x_col = f"{prefix}x"
        y_col = f"{prefix}y"
        z_col = f"{prefix}z"
        x_proj, y_proj = zip(*df[[x_col, y_col, z_col]].apply(
            lambda row: project_single_point(row[x_col], row[y_col], row[z_col]), axis=1))
        projected_data[f"{prefix}x"] = x_proj
        projected_data[f"{prefix}y"] = y_proj

    return pd.DataFrame(projected_data)

def plot_plane(ax, point, normal_vec, xlim, ylim):

    x = np.linspace(xlim[0], xlim[1], 20)
    y = np.linspace(ylim[0], ylim[1], 20)
    X, Y = np.meshgrid(x, y)


    if np.isclose(normal_vec[2], 0):
        Z = np.full_like(X, point[2])  # zを一定値に固定する
    else:
        # 平面の方程式からz座標を計算
        d = -np.dot(normal_vec, point)
        Z = (-normal_vec[0] * X - normal_vec[1] * Y - d) / normal_vec[2]

    return ax.plot_surface(X, Y, Z, alpha=0.5, color="green", edgecolor="none")

# データの準備
csv_file_path = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\dataset_margemag_tewnty.csv"
df_moto = pd.read_csv(csv_file_path)
df_moto = df_moto.drop(['time', 'rotate1', 'rotate2', 'rotate3', 'rotate4', 'force1', 'force2', 'force3', 'force4'], axis=1)
df_moto = df_moto.drop(['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9'], axis=1)
csv_file_path = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\output.csv"
df_ana = pd.read_csv(csv_file_path)


row1 = df_moto.iloc[50]
# 3Dプロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.set_zlim(-200, 0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plane = None
scatter_moto, = ax.plot([], [], [], 'ro', color = 'r', label='推定前', markersize=8)  # 点
line_moto, = ax.plot([], [], [], 'r-', color = 'r')  # 線


scatter_ana, = ax.plot([], [], [], 'ro', color = 'b', label='推定後', markersize=8)
line_ana, = ax.plot([], [], [], 'r-',  color = 'b')  # 1本目の線
# アニメーションの更新関数
def update(frame):
    global plane
    if plane:
        plane.remove()

    row = df_moto.iloc[frame]
    points = [
        (0, 0, 0),
        (row['Mc2x'], row['Mc2y'], row['Mc2z']),
    ]
    
    point1 = [row1['Mc2x'], row1['Mc2y'], row1['Mc2z']]
    point2 = [row1['Mc5x'], row1['Mc5y'], row1['Mc5z']]

    vec1 = np.array([point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]])
    vec2 = np.array([0,0, + 1])
    normal_vec = calculate_normal_vector(vec1, vec2)

    # 平面を描画
    plane = plot_plane(ax, point1, normal_vec, xlim=(-150, 150), ylim=(-150, 150))

    # データの点を描画
    x_moto, y_moto, z_moto = zip(*points)
    ax.plot(x_moto, y_moto, z_moto, 'r-', label="推定前")

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


    return plane

# アニメーションの設定
ani = FuncAnimation(fig, update, frames=len(df_moto), interval=100, blit=False)

plt.legend()
plt.show()
