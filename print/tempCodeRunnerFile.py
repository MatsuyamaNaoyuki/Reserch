import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ===== 設定 =====
PATH = r"C:\Users\shigf\Program\Reserch\0818_tubefinger_kijun_rere20250820_173021.pickle"
# KEYS = ["rotate", "force"] 
PATH = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortestnew.pickle"
# KEYS = ["rotate"] 
KEYS = ["sensor"] 
# KEYS = ["Mc5"] 
WINDOW = 5000          # 1画面で表示する点数
STEP   = 5000           # ←→・ホイールで動くステップ幅（点）
# =================
 # どれかを含む列を拾う


# データ読み込み
df = pd.read_pickle(PATH)
df = pd.DataFrame(df).reset_index(drop=True)  # インデックスは0,1,2,...にする

COLUMNS_TO_PLOT = [c for c in df.columns
                   if any(k.lower() in str(c).lower() for k in KEYS)]
COLUMNS_TO_PLOT.sort()

# 存在チェック
for col in COLUMNS_TO_PLOT:
    if col not in df.columns:
        raise KeyError(f"列が見つかりません: {col}. 利用可能: {list(df.columns)}")

n = len(df)
if n == 0:
    raise ValueError("データが空です。")


# 数値化（文字列が混じっていてもNaNにして続行）
df[COLUMNS_TO_PLOT] = df[COLUMNS_TO_PLOT].apply(pd.to_numeric, errors='coerce')

# 列ごとに z-score 標準化: (x - mean) / std
mu = df[COLUMNS_TO_PLOT].mean()
sigma = df[COLUMNS_TO_PLOT].std(ddof=0)        # 母標準偏差（0）で安定化
sigma = sigma.replace(0, np.nan)               # 定数列はゼロ割防止
df_std = (df[COLUMNS_TO_PLOT] - mu) / sigma
df_std = df_std.fillna(0)                      # ゼロ割や全NaN区間は0で埋める

# 以降、描画はこの SRC を参照（元dfは変更しない）
SRC = df_std


# ウィンドウ長の調整
WINDOW = min(WINDOW, n)
max_start = max(0, n - WINDOW)