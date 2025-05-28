import pandas as pd
import matplotlib.pyplot as plt
# from myclass import myfunction

# サンプルデータフレーム


df = pd.read_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\mixhit_fortest20250227_135315.pickle")
df = pd.DataFrame(df)
df = df.reset_index()
print(df.columns)   
columns_to_plot = ["rotate3","force3"]  # 指定したい列のリスト
df = df[:1000]
# 各列を個別にプロット
plt.figure()  # 新しい図を作成
for column in columns_to_plot:
    plt.plot(df.index, df[column], label=column)  # 列ごとにプロット

plt.title("Selected Columns")
plt.xlabel("Row Index")
plt.ylabel("Value")
plt.legend()  # 凡例を追加
plt.grid(True)
plt.show()