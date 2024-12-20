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

def plot_plane(ax, point, normal_vec, xlim, ylim, zlim=(-200, 0)):
    normal_vec = np.array(normal_vec)
    nx, ny, nz = normal_vec
    x0, y0, z0 = point

    # nx(X - x0) + ny(Y - y0) + nz(Z - z0) = 0
    # => nz(Z - z0) = -nx(X - x0) - ny(Y - y0)
    # => Z = z0 - (nx/nz)(X - x0) - (ny/nz)(Y - y0)  (if nz != 0)

    if np.isclose(nz, 0):
        # nz=0の場合、平面は垂直方向に無限に伸びるため、Zを独立変数としてもY, Xを明示的にZからは求められない。
        # ここでは、XとZをパラメータとして取り、そこからYを求める、またはYとZをパラメータとするなどの方法を取る。

        # 平面方程式: nx(X - x0) + ny(Y - y0) = 0
        # ny != 0の場合は、Y = y0 - (nx/ny)(X - x0)
        # ny = 0の場合は、nx(X - x0)=0 => X = x0で固定
        # それぞれに応じてパラメトリックに定義する

        if np.isclose(ny, 0):
            # この場合、ny=0なのでnx != 0が必須。よってX = x0で、YとZを独立変数にする。
            Y = np.linspace(ylim[0], ylim[1], 20)
            Z = np.linspace(zlim[0], zlim[1], 20)
            Y, Z = np.meshgrid(Y, Z)
            X = np.full_like(Y, x0)
        else:
            # ny != 0 の場合、XとZを独立変数として、Yを計算する。
            X = np.linspace(xlim[0], xlim[1], 20)
            Z = np.linspace(zlim[0], zlim[1], 20)
            X, Z = np.meshgrid(X, Z)
            Y = y0 - (nx/ny)*(X - x0)

        return ax.plot_surface(X, Y, Z, alpha=0.5, color="green", edgecolor="none")

    else:
        # nz != 0 の通常ケース。Zを(X,Y)から求める。
        x = np.linspace(xlim[0], xlim[1], 20)
        y = np.linspace(ylim[0], ylim[1], 20)
        X, Y = np.meshgrid(x, y)

        d = - (nx*x0 + ny*y0 + nz*z0)
        # Z = (-nx*X - ny*Y - d) / nz
        Z = (-nx*X - ny*Y - d) / nz
        return ax.plot_surface(X, Y, Z, alpha=0.5, color="green", edgecolor="none")


def to_unit_vector(vector):

    vector = np.array(vector)  # 入力をNumPy配列に変換
    norm = np.linalg.norm(vector)  # ベクトルのノルム（長さ）を計算
    
    if norm == 0:
        raise ValueError("ゼロベクトルには単位ベクトルを定義できません。")
    
    return vector / norm


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



    normal_vec = to_unit_vector([1,1,0])

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
