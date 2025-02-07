import csv, pickle
import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myclass import myfunction

def print_data_graph(df, columns_to_plot, titlename):
    plt.figure()  # 新しい図を作成
    for column in columns_to_plot:
        plt.plot(df.index, df[column], label=column)  # 列ごとにプロット

    plt.title(titlename)
    plt.xlabel("Row Index")
    plt.ylabel("Value")
    plt.legend()  # 凡例を追加
    plt.grid(True)
    plt.show()



df = pd.read_pickle(r"C:\Users\shigf\Program\data\sentan_test\modifydata_test20250127_184925.pickle")

# df = pd.DataFrame(datadf)

# グラフにしたい列を指定
columns_to_plot = ['force1', 'force2', 'force3', 'force4']  # 指定したい列のリスト


print_data_graph(df, columns_to_plot, titlename="magsensor")