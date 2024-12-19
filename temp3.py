import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def calculate_normal_vector(vec1, vec2):
    # 外積（クロス積）を計算
    normal_vector = np.cross(vec1, vec2)

    # 正規化（単位ベクトルにする）
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector


def rotate_point(x, y, angle_degrees):

    # 角度をラジアンに変換
    angle_radians = np.radians(angle_degrees)

    # 回転行列を定義
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])

    # 元の点をベクトルとして定義
    original_point = np.array([x, y])

    # 回転行列を適用
    rotated_point = np.dot(rotation_matrix, original_point)

    # 回転後の座標を返す
    return rotated_point[0], rotated_point[1]


def project_points_onto_plane(df, point1, point2):
    """
    z軸に平行な平面に点を投影する。
    """

    # 平面の法線ベクトルを計算 (z軸も考慮)
    vec1 = np.array([point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]])
    vec2 = np.array([point2[0], point2[1], point2[2] + 1])
    norm_vec = calculate_normal_vector(vec1, vec2)

    # 投影を計算する関数
    def project_single_point(x, y, z):
        targetvec = np.array([x,y,z])
        B = np.dot(norm_vec, targetvec) * norm_vec
        projected_point = targetvec- B


        # # 入力点から平面上の基準点へのベクトル
        # relative_vec = np.array([x - point1[0], y - point1[1], z - point1[2]])
        # # 投影計算
        # projection_length = np.dot(relative_vec, norm_vec)  # 法線方向への射影
        # projection_vec = projection_length * norm_vec       # 平面上の射影ベクトル
        # # 投影後の座標
        # projected_point = np.array([x, y, z]) - projection_vec

        projected_point[0], projected_point[1] = rotate_point(projected_point[0], projected_point[1], 180)

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

    # 投影後のデータフレームを作成
    return pd.DataFrame(projected_data)

# データの準備
csv_file_path = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\dataset_margemag_tewnty.csv"
df_moto = pd.read_csv(csv_file_path)

csv_file_path = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\output.csv"
df_ana = pd.read_csv(csv_file_path)

df_moto = df_moto.drop(['time', 'rotate1', 'rotate2', 'rotate3', 'rotate4', 'force1', 'force2', 'force3', 'force4'], axis=1)
df_moto = df_moto.drop(['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9'], axis=1)

row = df_moto.iloc[20]

points_moto1 = [row['Mc2x'], row['Mc2y'], row['Mc2z']]
points_moto2 = [row['Mc5x'], row['Mc5y'], row['Mc5z']]
df_moto = project_points_onto_plane(df_moto, points_moto1 ,points_moto2 )
df_ana = project_points_onto_plane(df_ana , points_moto1 ,points_moto2 )

print(df_ana)

fig, ax = plt.subplots()

# 軸の範囲を設定
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 初期化用のデータ（空のプロットを用意）
scatter_moto, = ax.plot([], [], 'ro', color='r', label='推定前')  # 点
line_moto, = ax.plot([], [], 'r-', color='r')  # 線

scatter_ana, = ax.plot([], [], 'ro', color='b', label='推定後')  # 点
line_ana, = ax.plot([], [], 'b-', color='b')  # 線

# アニメーションの更新関数
def update(frame):
    # フレームに応じたデータを取得
    row = df_moto.iloc[frame]
    points_moto = [
        (0, 0),
        (row['Mc2x'], row['Mc2y']),
        (row['Mc3x'], row['Mc3y']),
        (row['Mc4x'], row['Mc4y']),
        (row['Mc5x'], row['Mc5y']),
    ]
    x_moto, y_moto = zip(*points_moto)

    row = df_ana.iloc[frame]
    points_ana = [
        (0, 0),
        (row['Mc2x'], row['Mc2y']),
        (row['Mc3x'], row['Mc3y']),
        (row['Mc4x'], row['Mc4y']),
        (row['Mc5x'], row['Mc5y']),
    ]
    x_ana, y_ana = zip(*points_ana)

    # 点と線を更新
    scatter_moto.set_data(x_moto, y_moto)
    line_moto.set_data(x_moto[1:], y_moto[1:])

    scatter_ana.set_data(x_ana, y_ana)
    line_ana.set_data(x_ana[1:], y_ana[1:])

    return scatter_moto, line_moto, scatter_ana, line_ana

# アニメーションの設定
ani = FuncAnimation(fig, update, frames=len(range(0, 59)), interval=100, blit=False)

plt.legend()
plt.show()