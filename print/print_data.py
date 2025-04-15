import csv, pickle
import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from myclass import myfunction

def print_data_graph(df, columns_to_plot, titlename):
    plt.figure()  # 新しい図を作成

    # 指定されたインデックス番号をカラム名に変換
    columns_to_plot = [df.columns[col] for col in columns_to_plot if col < len(df.columns)]  # 範囲外を防ぐ

    for column in columns_to_plot:
        plt.plot(df.index, df[column], label=f"Column {column}")  # カラム名をラベルにする

    plt.title(titlename)
    plt.xlabel("Row Index")
    plt.ylabel("Value")
    plt.legend()  # 凡例を追加
    plt.grid(True)
    plt.show()



df = pd.read_pickle(r"C:\Users\WRS\Desktop\Matsuyama\Reserch\tube_softfinger_test_10_20250415_172035.pickle")

df = pd.DataFrame(df)
# front =int( len(df) / 3000 * 1)
# back =int( len(df) / 3000 * 15)
# print(front)
# print(back)
# df = df.iloc[front:back]
# # df = df.iloc[int(57500):int( 60000)]
# print(df.index)


columns_to_plot = [1,2,3,4]  # 指定したい列のリスト

    
print_data_graph(df, columns_to_plot, titlename="magsensor")