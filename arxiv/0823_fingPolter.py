import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider
from scipy.signal import find_peaks
from matplotlib.collections import LineCollection

def get_minimum(x, distance=150, prominence=5):
    # 1) 極小の検出
    minima, _ = find_peaks(-x, distance=distance, prominence=prominence)
    minima = np.sort(minima)
    if len(minima) < 2:
        return minima.astype(int)

    # 2) ロバストに基準間隔を推定（IQRで外れ値を除外）
    diffs = np.diff(minima)
    q1, q3 = np.percentile(diffs, [25, 75])
    mask = (diffs >= q1) & (diffs <= q3)
    base_interval = int(np.median(diffs[mask])) if mask.any() else int(np.median(diffs))
    base_interval = max(base_interval, 1)

    # 3) 区間ごとに gap を N 等分して補間（端点に整合）
    corrected = [int(minima[0])]
    for i in range(len(minima) - 1):
        left = int(minima[i])
        right = int(minima[i+1])
        gap = right - left

        # その区間に理論的に何本入るか（≒ round(gap/base)）
        N = int(np.round(gap / base_interval))
        N = max(N, 1)  # 少なくとも1区間

        if N == 1:
            # 何も欠けていないので右端だけ追加
            corrected.append(right)
        else:
            # gap を N 等分 → 内点を N-1 個入れる（端点のちょうど間）
            for k in range(1, N):
                # 等分の比率できっちり中央に来るように丸める
                pos = left + int(np.round(k * gap / N))
                corrected.append(pos)
            corrected.append(right)

    # 4) 整理（昇順・重複排除）
    minima_corrected = np.unique(np.array(corrected, dtype=int))
    return minima_corrected





smalldata = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit1500kaifortrain.pickle")
bigdata = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\mixhit1500kaifortrain.pickle")
difdata = smalldata - bigdata 
smalldata.columns = ["small" + col for col in smalldata.columns]
bigdata.columns = ["big" + col for col in bigdata.columns]
df = pd.concat([smalldata, bigdata], axis=1)

smallx = smalldata["smallrotate3"].values

smallminilen = get_minimum(smallx)


bigx = bigdata["bigrotate3"].values

bigminlen = get_minimum(bigx)
remove_vals = [573810, 72194, 72450, 573556, 573850, 552812, 552543,9678, 9399, 510778, 511055,51529, 51218]
bigminlen = bigminlen[~np.isin(bigminlen, remove_vals)]
add_vals = [72608, 72271,573972, 573636, 552611, 552946, 9451, 9790, 510832, 511166, 51570, 51236]
bigminlen = np.append(bigminlen, add_vals)
bigminlen = np.sort(bigminlen)

diffs = np.diff(bigminlen)

remove_vals = [585038]
smallminilen = smallminilen[~np.isin(smallminilen, remove_vals)]
add_vals = [584996]
smallminilen= np.append(smallminilen, add_vals)
smallminilen = np.sort(smallminilen)


print(len(smallminilen))
# 最小と最大のインデックス
min_idx = np.argmin(diffs)
max_idx = np.argmax(diffs)

print("間隔の最小値:", diffs[min_idx],
      " -> 区間:", smallminilen[min_idx], "～", smallminilen[min_idx+1])

print("間隔の最大値:", diffs[max_idx],
      " -> 区間:", smallminilen[max_idx], "～", smallminilen[max_idx+1])


# df = difdata 



# ===== 設定 =====
PATH = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\0818_tubefinger_kijun_rere20250820_173021.pickle"
KEYS = ["smallrotate"] 
# KEYS = ["sensor"] 
# KEYS = ["Mc"] 
WINDOW = 5000          # 1画面で表示する点数
STEP   = 5000           # ←→・ホイールで動くステップ幅（点）
# =================
 # どれかを含む列を拾う


# # データ読み込み
# df = pd.read_pickle(PATH)
# df = pd.DataFrame(df).reset_index(drop=True)  # インデックスは0,1,2,...にする

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
# mu = df[COLUMNS_TO_PLOT].mean()
# sigma = df[COLUMNS_TO_PLOT].std(ddof=0)        # 母標準偏差（0）で安定化
# sigma = sigma.replace(0, np.nan)               # 定数列はゼロ割防止
# df_std = (df[COLUMNS_TO_PLOT] - mu) / sigma
# df_std = df_std.fillna(0)                      # ゼロ割や全NaN区間は0で埋める

# # 以降、描画はこの SRC を参照（元dfは変更しない）
# SRC = df_std

SRC = df
# ウィンドウ長の調整
WINDOW = min(WINDOW, n)
max_start = max(0, n - WINDOW)

# 図の作成
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.15)  # スライダー用に下を少し空ける

# 初期データ（先頭から）
start = 0
end = start + WINDOW
x = np.arange(start, end)

lines = {}
for col in COLUMNS_TO_PLOT:
    y = SRC[col].iloc[start:end].to_numpy()
    (line,) = ax.plot(x, y, label=col)
    lines[col] = line


minima_idx = np.asarray(smallminilen)  # 1D
vlines = LineCollection([], colors='red', linewidths=1.2, alpha=0.9)
ax.add_collection(vlines)


ax.set_title("Selected Columns (scroll with slider / arrow keys / wheel)")
ax.set_xlabel("Row Index")
ax.set_ylabel("Value")
ax.grid(True)
ax.legend()

def update_ylim(s, e):
    ymin = np.inf
    ymax = -np.inf
    for col in COLUMNS_TO_PLOT:
        yy = SRC[col].iloc[s:e].to_numpy()
        if np.all(np.isnan(yy)):
            continue
        ymin = min(ymin, np.nanmin(yy))
        ymax = max(ymax, np.nanmax(yy))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ymin, ymax = -1.0, 1.0
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

def redraw(start_idx: int):
    start_idx = int(max(0, min(start_idx, max_start)))
    end_idx = start_idx + WINDOW
    x = np.arange(start_idx, end_idx)

    # データ線を更新
    for col, line in lines.items():
        y = SRC[col].iloc[start_idx:end_idx].to_numpy()
        line.set_data(x, y)

    # 軸範囲更新
    ax.set_xlim(x[0], x[-1])
    update_ylim(start_idx, end_idx)

    # ★ この表示範囲に入る極小値だけ縦線を出す
    y0, y1 = ax.get_ylim()
    hit = (minima_idx >= start_idx) & (minima_idx < end_idx)
    mins_in_window = minima_idx[hit]

    # LineCollection 用のセグメント [( (x,y0),(x,y1) ), ...] を作成
    segments = [((int(i), y0), (int(i), y1)) for i in mins_in_window]
    vlines.set_segments(segments)

    fig.canvas.draw_idle()
    return start_idx

# スライダー
ax_slider = plt.axes([0.1, 0.05, 0.8, 0.04])
slider = Slider(ax_slider, "Start", 0, max_start, valinit=0, valstep=1)

def on_slider(val):
    redraw(val)
slider.on_changed(on_slider)

# キー操作（←→でSTEPずつ移動）
def on_key(event):
    cur = int(slider.val)
    if event.key == "left":
        slider.set_val(cur - STEP)
    elif event.key == "right":
        slider.set_val(cur + STEP)
fig.canvas.mpl_connect("key_press_event", on_key)

# マウスホイールでも移動（上=戻る/下=進む）
def on_scroll(event):
    cur = int(slider.val)
    if event.button == "up":
        slider.set_val(cur - STEP)
    elif event.button == "down":
        slider.set_val(cur + STEP)
fig.canvas.mpl_connect("scroll_event", on_scroll)

# 初期描画
update_ylim(start, end)
plt.show()
