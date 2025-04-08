import pandas as pd
import matplotlib.pyplot as plt
# from myclass import myfunction

# サンプルデータフレーム


df = pd.read_pickle(r"C:\Users\shigf\Program\data\withhit\testyou\withhit_continumtest20250227_132458.pickle")
df = pd.DataFrame(df)

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