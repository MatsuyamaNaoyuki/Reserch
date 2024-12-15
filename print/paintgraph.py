import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import time

# CSVファイルのパスを指定
csv_file_path = 'Autaum_mid_data_small_double20241017_214008.csv'

# CSVファイルを読み込む
df = pd.read_csv(csv_file_path, header=None)


# df = df.drop(254)

# df = df.drop(196)
# df = df.drop(0)


# # インデックスをリセット
df = df.reset_index(drop=True)

# 読み込んだデータの確認（最初の5行を表示）
print(df.head())

first_time = df[0][0]
first_time = datetime.strptime(first_time, "%Y-%m-%d %H:%M:%S.%f")


for i in range(len(df)):
    tmptime = datetime.strptime(df[0][i],  "%Y-%m-%d %H:%M:%S.%f")
    seconds = tmptime - first_time
    milliseconds = seconds.total_seconds()
    df.loc[i, 0] = milliseconds
    # df.loc[i, 5] = df.loc[i, 5] - 466
    # df.loc[i, 6] = df.loc[i, 6] - 546.0
    # df.loc[i, 7] = df.loc[i, 7] - 465.0
    # df.loc[i, 9] = df.loc[i, 9] * 5 / 1024
    # df.loc[i, 10] = df.loc[i, 10] * 5 / 1024
    # df.loc[i, 11] = df.loc[i, 11] * 5 / 1024
    # df.loc[i, 12] = df.loc[i, 12] * 5 / 1024
    # df.loc[i, 13] = df.loc[i, 13] * 5 / 1024
    # df.loc[i, 14] = df.loc[i, 14] * 5 / 1024
    # df.loc[i, 15] = df.loc[i, 15] * 5 / 1024
    # df.loc[i, 16] = df.loc[i, 16] * 5 / 1024
    # df.loc[i, 17] = df.loc[i, 17] * 5 / 1024
    



# 列ごとの最大値を取得
max_values = df.max()

# 列ごとの最小値を取得
min_values = df.min()

print(max_values, min_values)
    
print(df)

# グラフを作成する列を指定（例として 'Column1' と 'Column2' を使用）
# 実際の列名に置き換えてください
# 例えば、1列目のデータをプロットする場合
# 複数の列をプロット（例として1列目、2列目、3列目をプロット）
x_values = df[0]  # 横軸はデータ数（インデックス）
# x_values = range(len(df)) 
y_values_1 = df[5]  # 1列目
y_values_2 = df[6]  # 2列目
y_values_3 = df[7]  # 3列目

# y_values_4 = df[12]  # 1列目
# y_values_5 = df[13]  # 2列目
# y_values_6 = df[14]  # 3列目

# y_values_7 = df[15]  # 1列目
# y_values_8 = df[16]  # 2列目
# y_values_9 = df[17]  # 3列目

# グラフを作成
plt.figure(figsize=(10, 6))

# 各列をプロット
plt.plot(x_values, y_values_1, label='right sensor values')
plt.plot(x_values, y_values_2, label='sentors sensor values')
plt.plot(x_values, y_values_3, label='left sensor values')
# plt.plot(x_values, y_values_4, label='sentor left values')
# plt.plot(x_values, y_values_5, label='sentor sentor values')
# plt.plot(x_values, y_values_6, label='sentor right values')
# plt.plot(x_values, y_values_7, label='root left values')
# plt.plot(x_values, y_values_8, label='root sentor values')
# plt.plot(x_values, y_values_9, label='root right values')

# グラフのタイトル、ラベル、凡例を設定
plt.title('root sensor Value', fontsize=24)
plt.xlabel('second', fontsize=24)
plt.ylabel('sensor Value', fontsize=24)

plt.ylim(539,881)


plt.tick_params(labelsize=18)
# 凡例を表示
plt.legend(fontsize = 15)

# グラフを表示
plt.show()