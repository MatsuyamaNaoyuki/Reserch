import pandas as pd
import matplotlib.pyplot as plt
from myclass import myfunction

# サンプルデータフレーム


df = pd.read_csv(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\sentan_morecam\modifydata20250122.csv")

columns_to_plot = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9']  # 指定したい列のリスト
df = df.iloc[:12000]
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