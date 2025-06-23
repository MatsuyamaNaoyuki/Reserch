import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
# from myclass import myfunction

df = pd.read_pickle(r"C:\Users\shigf\Program\data\withhit\with_hit_for_test\with_hit_for_test20250226_133722.pickle")
df = pd.DataFrame(df)
df = df.reset_index()
print(df.columns)   
columns_to_plot = ["rotate1","rotate2", "rotate3", "rotate4"]  # 指定したい列のリスト
df = df[:650]
# 各列を個別にプロット

labelname = ["motor1", "motor2", "motor3", "motor4"]
plt.figure(figsize=(12, 2))

for i, column in enumerate(columns_to_plot):
    plt.plot(df.index, df[column], label=labelname[i])  # 列ごとにプロット

plt.title("motor angle")
plt.xlabel("Time step (index)")
plt.ylabel("Motor angle [°]")
plt.legend()  # 凡例を追加
plt.grid(True)
plt.tight_layout()
plt.show()