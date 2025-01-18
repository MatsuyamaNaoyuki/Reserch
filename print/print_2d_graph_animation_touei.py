import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def calculate_normal_vector(vec):
    # 任意のベクトルを受け取り、正規化する関数
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("法線ベクトルがゼロベクトルです。")
    return vec / norm

def rotate_point(x, y, angle_degrees):
    # 2D平面上で点(x,y)をangle_degrees度回転する
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])
    original_point = np.array([x, y])
    rotated_point = np.dot(rotation_matrix, original_point)
    return rotated_point[0], rotated_point[1]

def project_points_onto_plane(df, normal_vec, point_on_plane=(0,0,0)):
    """
    任意の法線ベクトルを持つ平面へ点群を投影する関数。
    point_on_plane: 平面上の1点 (x0, y0, z0)
    normal_vec: 平面の法線ベクトル（正規化前でも可）
    """

    norm_vec = calculate_normal_vector(normal_vec)
    point_on_plane = np.array(point_on_plane)

    # 投影を計算する内部関数
    def project_single_point(x, y, z):
        target_vec = np.array([x, y, z])
        relative_vec = target_vec - point_on_plane
        dist_along_normal = np.dot(relative_vec, norm_vec)
        projected_point = target_vec - dist_along_normal * norm_vec
        # 2D平面表示用に回転
        px, py = rotate_point(projected_point[0], projected_point[1], 180)
        return px, py

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

# データの準備
csv_file_path = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\dataset_margemag_tewnty.csv"
df_moto = pd.read_csv(csv_file_path)

csv_file_path = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\output.csv"
df_ana = pd.read_csv(csv_file_path)

df_moto = df_moto.drop(['time', 'rotate1', 'rotate2', 'rotate3', 'rotate4', 'force1', 'force2', 'force3', 'force4'], axis=1)
df_moto = df_moto.drop(['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9'], axis=1)

# 例としてdf_motoの50行目のマーカーを利用して平面上の点を決めるなど、必要に応じてpoint_on_planeを定義します。
row = df_moto.iloc[50]
points_moto1 = [0,0,0]  # 平面上の基準点として使用
# 法線ベクトルを[-1,1,0]とする
normal_vec = [1,1,0]

df_moto_proj = project_points_onto_plane(df_moto, normal_vec, points_moto1)
df_ana_proj = project_points_onto_plane(df_ana, normal_vec, points_moto1)

print(df_ana_proj)

fig, ax = plt.subplots()
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.set_xlabel('X')
ax.set_ylabel('Y')

scatter_moto, = ax.plot([], [], 'ro', color='r', label='推定前')  # 点
line_moto, = ax.plot([], [], 'r-', color='r')  # 線

scatter_ana, = ax.plot([], [], 'ro', color='b', label='推定後')  # 点
line_ana, = ax.plot([], [], 'b-', color='b')  # 線

def update(frame):
    row_moto = df_moto_proj.iloc[frame]
    points_moto = [
        (0, 0),
        (row_moto['Mc2x'], row_moto['Mc2y']),
        (row_moto['Mc3x'], row_moto['Mc3y']),
        (row_moto['Mc4x'], row_moto['Mc4y']),
        (row_moto['Mc5x'], row_moto['Mc5y']),
    ]
    x_moto, y_moto = zip(*points_moto)

    row_ana = df_ana_proj.iloc[frame]
    points_ana = [
        (0, 0),
        (row_ana['Mc2x'], row_ana['Mc2y']),
        (row_ana['Mc3x'], row_ana['Mc3y']),
        (row_ana['Mc4x'], row_ana['Mc4y']),
        (row_ana['Mc5x'], row_ana['Mc5y']),
    ]
    x_ana, y_ana = zip(*points_ana)

    scatter_moto.set_data(x_moto, y_moto)
    line_moto.set_data(x_moto[1:], y_moto[1:])

    scatter_ana.set_data(x_ana, y_ana)
    line_ana.set_data(x_ana[1:], y_ana[1:])

    return scatter_moto, line_moto, scatter_ana, line_ana

ani = FuncAnimation(fig, update, frames=len(range(0, 100)), interval=100, blit=False)
plt.legend()
plt.show()
