import pandas as pd
import matplotlib.pyplot as plt
# from myclass import myfunction

# サンプルデータフレーム


df = pd.read_pickle(r"C:\Users\shigf\Program\data\0519_mech-newfinger\withhittest\0519_mech-newfinger_withhittest__20250523_180608.pickle")
df = pd.DataFrame(df)
print(df.columns)   
columns_to_plot = ["sensor7","sensor8","sensor9"]  # 指定したい列のリスト
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