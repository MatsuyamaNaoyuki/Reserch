import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

def replace_short_neg_sequences(series, max_value, min_value):
    limit_length=8
    values = series.values
    start = None  # 連続部分の開始位置を記録

    for i in range(len(values)):
        if values[i] == min_value:

            if start is None:
                # print(f"i = {i}")
                start = i  # 連続の開始位置
        else:
            if start is not None:
                # print(f"start = {start}")
                # print(f"i = {i}")
                length = i - start
                # print(f"length = {length}")
                if length <= limit_length:
                    # print(values[start - 3:i + 3])
                    values[start:i] = max_value  # 置き換え
                    # print(values[start - 3:i + 3])
                start = None  # 次の連続部分を探す
                # print("---------------------------------------")
            

    # 最後の部分の処理（連続が終わらない場合）
    if start is not None:
        length = len(values) - start
        if length <= limit_length:
            values[start:] = max_value

    return pd.Series(values, index=series.index)




def change_force_to_2_value(series):
    min_value = -200
    max_value = 0
    window_size = 10
    smoothed_col = series.rolling(window=window_size, min_periods=1).mean()
    diff = smoothed_col.sub(smoothed_col.shift(2)).abs()
    two_val = diff.apply(lambda x: max_value if x >= 4 else min_value)
    two_val = replace_short_neg_sequences(two_val, max_value, min_value)
    two_val = replace_short_neg_sequences(two_val, min_value, max_value)
    return two_val, diff
    

# データの読み込み
motor = pd.read_pickle(r"C:\Users\shigf\Program\Reserch\motor20250223_021713.pickle")
motor = pd.DataFrame(motor)
print(motor)
print(motor.iloc[:, 9])
motor.loc[:, "Flag"], motor.loc[:, "diff"]  = change_force_to_2_value(motor.iloc[:, 9])

motor = motor.iloc[1500:2500]


# 平滑化

# プロット
columns_to_plot = [9, "Flag", "diff"]
plt.figure()
for column in columns_to_plot:
    plt.plot(motor.index, motor[column], label=column)

plt.title("Selected Columns")
plt.xlabel("Row Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

print(motor.iloc[2])