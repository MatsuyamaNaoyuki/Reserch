from myclass import myfunction
import matplotlib.pyplot as plt
import pandas as pd

df = myfunction.load_pickle(r"C:\Users\shigf\Desktop\5月15日のごみ箱\mixhit_300020250225_204120.pickle")

df = pd.DataFrame(df)
l = int(len(df) / 2)


df1 = df[97822:97822 +700]

print(df.columns)   
columns_to_plot = ["rotate1", "rotate2", "rotate3"]  # 指定したい列のリスト

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