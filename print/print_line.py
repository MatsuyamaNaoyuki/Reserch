import pickle 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



def make_line_graph(line1, line2=None):
    """
    3Dプロットを描画する関数。
    
    Parameters:
        line1: pandas.DataFrame（必須） → メインのデータ
        line2: pandas.DataFrame（オプション, デフォルト: None） → 比較用のデータ（青）
    """

    # 基準点
    xbase, ybase, zbase = 14.19 + 12.59, -10.26 - 16.23, 185.34 - 198.79

    def extract_coordinates(line):
        """各点群のx, y, z座標をリストで取得"""
        x = [line[f'Mc{i}x'].tolist() for i in range(2, 6)]
        y = [line[f'Mc{i}y'].tolist() for i in range(2, 6)]
        z = [line[f'Mc{i}z'].tolist() for i in range(2, 6)]
        return x, y, z

    # line1 のデータ
    x1, y1, z1 = extract_coordinates(line1)

    # line2 が存在する場合
    x2, y2, z2 = extract_coordinates(line2) if line2 is not None else (None, None, None)

    # 3Dプロットの作成
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_data(ax, x, y, z, color):
        """点群と線をプロットする関数"""
        for i in range(len(line1)):  # 各行ごとにプロット
            ax.scatter(x[0][i], y[0][i], z[0][i], c=color, s=5)
            ax.scatter(x[1][i], y[1][i], z[1][i], c=color, s=5)
            ax.scatter(x[2][i], y[2][i], z[2][i], c=color, s=5)
            ax.scatter(x[3][i], y[3][i], z[3][i], c=color, s=5)

            # 基準点と各データを線でつなぐ
            ax.plot(
                [xbase, x[0][i], x[1][i], x[2][i], x[3][i]],
                [ybase, y[0][i], y[1][i], y[2][i], y[3][i]],
                [zbase, z[0][i], z[1][i], z[2][i], z[3][i]],
                c=color
            )
    if line2 is not None:
        plot_data(ax, x2, y2, z2, 'b')
        ax.plot([], [], [], c='b', label="Actual") 

    # line1（赤）
    plot_data(ax, x1, y1, z1, 'r')
    ax.plot([], [], [], c='r', label="Estimated")

    # line2（青）→ もし `None` でなければ描画
    
    # 軸のラベル
    # ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    # X軸の目盛りを消す
    ax.set_xticklabels([])

    # 軸の範囲を統一
    all_x = [xbase] + x1[0] + x1[1] + x1[2] + x1[3]
    all_y = [ybase] + y1[0] + y1[1] + y1[2] + y1[3]
    all_z = [zbase] + z1[0] + z1[1] + z1[2] + z1[3]

    if line2 is not None:
        all_x += x2[0] + x2[1] + x2[2] + x2[3]
        all_y += y2[0] + y2[1] + y2[2] + y2[3]
        all_z += z2[0] + z2[1] + z2[2] + z2[3]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x, mid_y, mid_z = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0, (z_max + z_min) / 2.0

    ax.set_box_aspect([1, 1, 1])  # 比率を1:1:1に設定
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend(fontsize=20)

    # グラフを表示
    plt.show()


# CSVファイルの読み込み
file_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\moredataset1000\output.pickle"
data = pd.read_pickle(file_path)
selected_columns = data[['Mc2x', 'Mc2y', 'Mc2z','Mc3x', 'Mc3y', 'Mc3z','Mc4x', 'Mc4y', 'Mc4z','Mc5x', 'Mc5y', 'Mc5z']]


select = [0, 8150, 8380, 8630]

output = selected_columns.iloc[select]


file_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\currentOK0203\currentOKtest_020320250203_201037.pickle"
data = pd.read_pickle(file_path)
selected_columns = data[['Mc2x', 'Mc2y', 'Mc2z','Mc3x', 'Mc3y', 'Mc3z','Mc4x', 'Mc4y', 'Mc4z','Mc5x', 'Mc5y', 'Mc5z']]
base = selected_columns.iloc[select]

make_line_graph(output, base)
# lendata = len(data)

# grid_size = 10.0  # 1.0の精度で丸める
# rounded_data = selected_columns.applymap(lambda x: round(x / grid_size) * grid_size)

# # ユニークなデータを抽出
# unique_data = rounded_data.drop_duplicates()
# print(unique_data)

# make_line_graph(unique_data)


