import sys,os, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting
import csv
import pprint
from myclass import myfunction

import pandas as pd

# 元データのパス
path = r"C:\Users\shigf\Program\data\0816tubefinger_re\0816_tubefinger_hit_1500kai_re20250816_134303.pickle"

# 読み込み（pickleがDataFrame以外でも来た時はDataFrame化を試みる）
obj = pd.read_pickle(path)
if isinstance(obj, pd.DataFrame):
    df = obj
else:
    df = pd.DataFrame(obj)

# NaN を含む行を検出
nan_rows_mask = df.isna().any(axis=1)
nan_count = int(nan_rows_mask.sum())

print(f"NaNを含む行数: {nan_count} / 総行数: {len(df)}")

# NaN を含む行を除去
df_clean = df.loc[~nan_rows_mask].copy()
print(f"NaN除去後の形状: {df_clean.shape}")

# 結果を表示（行数が多い場合は長く表示されます）
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
    print(df_clean)

# 必要なら保存（コメント解除）
# out_path = path.replace(".pickle", "_noNaN.pickle")
# df_clean.to_pickle(out_path)
# print(f"保存しました: {out_path}")
