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



df = pd.read_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_withhit\withhit_fortest20250227_134803.pickle")

df = pd.DataFrame(df)
front =int( len(df) / 5 * 3)
back =int( len(df) / 5 * 4)
print(front)
print(back)
df = df.iloc[front:back]
# df = df.iloc[int(57500):int( 60000)]
print(df.index)
columns_to_plot = ['rotate1', 'rotate2', 'rotate3', 'rotate4']  # 指定したい列のリスト


print_data_graph(df, columns_to_plot, titlename="magsensor")