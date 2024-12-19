# import csv
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# import math
from scipy.spatial.transform import Rotation as R
# import japanize_matplotlib


# def find_plane_and_rotate_to_xz(points):
#     # 1. 平面の推定
#     # 平面は Ax + By + Cz + D = 0 の形式で表現される
#     # ここでは最小二乗法で平面を推定する
#     points = np.array(points)
#     centroid = np.mean(points, axis=0)  # 中心点を計算
#     normalized_points = points - centroid  # 中心点を原点とする

#     # SVDを使用して主軸を求める
#     _, _, vh = np.linalg.svd(normalized_points)
#     normal_vector = vh[-1]  # 最小固有値に対応する固有ベクトルが法線ベクトル

#     # 2. 回転行列の計算
#     # 法線ベクトルを z 軸方向に揃える回転行列を計算
#     z_axis = np.array([0, 0, 1])
#     rotation_axis = np.cross(normal_vector, z_axis)
#     rotation_axis_norm = np.linalg.norm(rotation_axis)

#     if rotation_axis_norm != 0:
#         rotation_axis = rotation_axis / rotation_axis_norm  # 回転軸を正規化
#         rotation_angle = np.arccos(np.dot(normal_vector, z_axis))
#         rotation_vector = rotation_axis * rotation_angle
#         rotation = R.from_rotvec(rotation_vector)  # 回転行列を作成
#     else:
#         # 法線がすでに z 軸方向に揃っている場合
#         rotation = R.from_rotvec([0, 0, 0])

#     # 3. 点を回転
#     rotated_points = rotation.apply(points)

#     return rotated_points, rotation

def rotate_to_xz(points):
    P = np.array(points)
    v1 = P[2] - P[1]
    v2 = P[3] - P[1]
    normal = np.cross(v1, v2)
    # 法線ベクトルを正規化
    normal = normal / np.linalg.norm(normal)

    # 目標とする法線方向は y 軸 (0,1,0)
    target = np.array([0.0, 1.0, 0.0])

    # 現在のnormalをtargetに回転させる回転行列を求める
    # 回転軸 = normal × target
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c = np.dot(normal, target)

    if s == 0:
        # すでにnormalが(0,1,0)か、(0,-1,0)方向に揃っている場合
        # normalが(0,1,0)なら回転不要、(0,-1,0)なら180度回転必要
        if c < 0:
            # 180度回転: x軸またはz軸回りの180度回転
            # ここではx軸回り180度回転とする(任意)
            R = np.array([[1, 0,  0],
                        [0, -1, 0],
                        [0, 0, -1]])
        else:
            # 回転不要
            R = np.eye(3)
    else:
        # ロドリゲスの回転公式を用いて回転行列を求める
        v = v / s  # 回転軸を正規化
        K = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        R = np.eye(3) + K * s + K @ K * (1 - c)


    return R


pd.options.display.float_format = '{:.2f}'.format


csv_file_path="C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\dataset_margemag_tewnty.csv"
df = pd.read_csv(csv_file_path)
df = df.drop(['time', 'rotate1', 'rotate2', 'rotate3', 'rotate4', 'force1', 'force2', 'force3', 'force4'], axis = 1)
df = df.drop(['sensor1','sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9',], axis = 1)




row = df.iloc[60]


points = [
    (0,0,0),
    (row['Mc2x'], row['Mc2y'], row['Mc2z']),
    (row['Mc3x'], row['Mc3y'], row['Mc3z']),
    (row['Mc4x'], row['Mc4y'], row['Mc4z']),
    (row['Mc5x'], row['Mc5y'], row['Mc5z']),
]


print(points)

# points0 = [
#     (row['Mc1x']-row['Mc1x'], row['Mc1y']-row['Mc1y'], row['Mc1z']-row['Mc1z']),
#     (row['Mc2x']-row['Mc1x'], row['Mc2y']-row['Mc1y'], row['Mc2z']-row['Mc1z']),
#     (row['Mc3x']-row['Mc1x'], row['Mc3y']-row['Mc1y'], row['Mc3z']-row['Mc1z']),
#     (row['Mc4x']-row['Mc1x'], row['Mc4y']-row['Mc1y'], row['Mc4z']-row['Mc1z']),
#     (row['Mc5x']-row['Mc1x'], row['Mc5y']-row['Mc1y'], row['Mc5z']-row['Mc1z']),
# ]
# rotation = rotate_to_xz(points[1:])
# P = np.array(points)
# rotated_points = (rotation @ P.T).T




# x0,y0,z0 = zip(*points0)

x, y, z = zip(*points)


rcolors = ['r', 'r', 'r', 'r', 'r']  # 各点の色
bcolors = ['b', 'b', 'b', 'b', 'b']  # 各点の色
labels = ['Point 1', 'Point 2', 'Point 3', 'Point 4', 'Point 5']  # 凡例用ラベル
# 3Dプロット
fig = plt.figure()

plt.rcParams['xtick.labelsize'] = 20  # x軸目盛りラベルのサイズ
plt.rcParams['ytick.labelsize'] = 20
ax = fig.add_subplot(111, projection='3d')

# 散布図を作成
for i in range(len(x)):
    ax.scatter(x[i], y[i], z[i], color=rcolors[i], s=80)  # sは点のサイズ


# for i in range(len(x0)):
#     # print(x0[i],y0[i],z0[i])
#     ax.scatter(x0[i], y0[i], z0[i], color=bcolors[i], s=80)  # sは点のサイズ

ax.plot(x[1:], y[1:], z[1:], color='r', label='補正前')
# ax.plot(x0[1:], y0[1:], z0[1:], color='b', label='補正後')
# 軸ラベルを設定
# ax.set_xlabel('X Label', fontsize = 12)
# ax.set_ylabel('Y Label', fontsize = 12)
# ax.set_zlabel('Z Label', fontsize = 12)
# xticklabels = ax.get_xticklabels()
# yticklabels = ax.get_yticklabels()
# zticklabels = ax.get_xticklabels()

# ax.set_xticklabels(xticklabels,fontsize=12)
# ax.set_yticklabels(yticklabels,fontsize=12)
# ax.set_zticklabels(yticklabels,fontsize=12)

# # ax.set_xlim(-125, 125)
# ax.set_ylim(-125, 125)
# ax.set_zlim(-250, 0)
ax.axis('equal')



ax.legend(fontsize=20) 

# 表示
plt.show()