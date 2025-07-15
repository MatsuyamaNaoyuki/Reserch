from myclass import myfunction
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


filename = r"C:\Users\shigf\Program\data\testyou.pickle"
df = myfunction.load_pickle(filename)

print(len(df))

df1 = df[:2500]
df2 = df[-2500:]

df_combined = pd.concat([df1, df2], axis=0).reset_index(drop=True)


# NaNにしたい行のインデックス
nan_rows = [3, 1000]

# 'type' 以外の列名リストを取得
cols_to_nan = df_combined.columns.difference(['type'])

# 指定した行・列だけNaNにする
df_combined.loc[nan_rows, cols_to_nan] = np.nan
print(df)
# myfunction.wirte_pkl(df_combined, r"C:\Users\shigf\Program\data\testnan")
