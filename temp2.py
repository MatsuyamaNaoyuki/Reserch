import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_plane(ax, point, normal_vec, xlim, ylim, zlim=(-200, 0)):
    normal_vec = np.array(normal_vec)
    nx, ny, nz = normal_vec
    x0, y0, z0 = point


    if np.isclose(nz, 0):

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
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("ゼロベクトルには単位ベクトルを定義できません。")
    return vector / norm

# データの準備

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

    normal_vec = to_unit_vector([-1,1,0])

    # 平面を描画
    plane = plot_plane(ax, (0,0,0), normal_vec, xlim=(-150, 150), ylim=(-150, 150))

    return plane,

# アニメーションの設定
ani = FuncAnimation(fig, update, frames=1000, interval=100, blit=False)

plt.legend()
plt.show()