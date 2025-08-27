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

def align_B_on_A_n_cycles(seriesA, seriesB, a_mins, b_mins, k, n_cycles=10):
    """
    seriesA: DataFrame の列 (A のデータ)
    seriesB: DataFrame の列 (B のデータ)
    a_mins, b_mins: 極小値のインデックス配列
    k: 開始する極小値の番号
    n_cycles: 何山分を表示するか
    """
    # n_cyclesぶん極小値を取る（例: 2山なら a0, a1, a2）
    a_start = int(a_mins[k])
    a_end   = int(a_mins[k+n_cycles])
    b_start = int(b_mins[k])
    b_end   = int(b_mins[k+n_cycles])

    x_idx = np.arange(a_start, a_end+1, dtype=int)
    A_win = seriesA.iloc[a_start:a_end+1].to_numpy()
    B_on_A_win = np.full_like(A_win, np.nan, dtype=float)

    # 各山ごとに対応付け
    for cycle in range(n_cycles):
        a0, a1 = int(a_mins[k+cycle]), int(a_mins[k+cycle+1])
        b0, b1 = int(b_mins[k+cycle]), int(b_mins[k+cycle+1])

        lenA = a1 - a0
        lenB = b1 - b0
        m = min(lenA, lenB)

        if m > 0:
            # A のインデックスに合わせて代入
            B_on_A_win[(a0-a_start):(a0-a_start)+m] = seriesB.iloc[b0:b0+m].to_numpy()

    return x_idx, A_win, B_on_A_win




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
# KEYS = ["force"] 
KEYS = ["sensor1","sensor2","sensor3"] 
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



pairs = []
for c in COLUMNS_TO_PLOT:
    if c.startswith("big"):
        tail = c[3:]
        a_col = "small" + tail
        if a_col in df.columns:
            pairs.append((a_col, c))

myfunction.print_val(pairs)
a_mins = smallminilen
b_mins = bigminlen
max_k = len(a_mins) - 3



# 図の作成
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.18)  # スライダー用に下を少し空ける


linesA = {}
linesB = {}

k0 = 0
x_idx0, _, _=align_B_on_A_n_cycles(df[pairs[0][0]], df[pairs[0][1]], a_mins, b_mins, k0)
for a_cols, b_cols in pairs:
    _, A_win0, B_win0 = align_B_on_A_n_cycles(df[a_cols], df[b_cols], a_mins, b_mins, k0)
    (lA,) = ax.plot(x_idx0, A_win0, label=a_cols, alpha=0.9)
    (lB,) = ax.plot(x_idx0, B_win0, label=b_cols, alpha=0.9)
    linesA[a_cols] = lA
    linesB[b_cols] = lB

ax.set_title("Two cycles aligned on A's minima (B is truncated/NaN-filled)")
ax.set_xlabel("Row Index (A)")
ax.set_ylabel("Value")
ax.grid(True)
ax.legend(ncol=2)



ymin, ymax = np.inf, -np.inf
for l in list(linesA.values()) + list(linesB.values()):
    x, y = l.get_data()
    if len(y):
        yv = np.array(y, float)
        yv = yv[~np.isnan(yv)]
        if yv.size:
            ymin = min(ymin, np.min(yv))
            ymax = max(ymax, np.max(yv))

if not np.isfinite(ymin):
    ymin, ymax = -1.0, 1.0

pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
ax.set_ylim(ymin - pad, ymax + pad)
# ===== ここから差し替え：スライダー2本 + 同期 =====

# 初期サイクル数
n_cycles = 10

def compute_max_k():
    # k..k+n_cycles が両方で有効になる上限
    mk = min(len(a_mins), len(b_mins)) - (n_cycles + 1)
    return max(0, mk)

max_k = compute_max_k()

def redraw_k(k:int):
    k = int(np.clip(k, 0, max_k))
    x_idx, _, _ = align_B_on_A_n_cycles(
        df[pairs[0][0]], df[pairs[0][1]],
        a_mins, b_mins, k, n_cycles=n_cycles
    )
    ax.set_xlim(x_idx[0], x_idx[-1])

    for a_col, b_col in pairs:
        x_idx, A_win, B_win = align_B_on_A_n_cycles(
            df[a_col], df[b_col],
            a_mins, b_mins, k, n_cycles=n_cycles
        )
        linesA[a_col].set_data(x_idx, A_win)
        linesB[b_col].set_data(x_idx, B_win)

    ax.set_title(
        f"Aligned on A's minima  |  k={k} → k+{n_cycles}={k+n_cycles}  "
        f"(A idx: {int(a_mins[k])}..{int(a_mins[k+n_cycles])})"
    )
    fig.canvas.draw_idle()

# --- スライダー領域 ---
ax_slider_k = plt.axes([0.12, 0.08, 0.76, 0.04])
ax_slider_n = plt.axes([0.12, 0.03, 0.76, 0.04])

s_k = Slider(ax_slider_k, "Start cycle k", 0, max_k, valinit=0, valstep=1)
s_n = Slider(ax_slider_n, "n_cycles", 1, 10, valinit=n_cycles, valstep=1)  # 1～10山で調整可

def refresh_slider_ranges():
    """n_cycles変更時に k の上限を更新し、はみ出しを補正"""
    global max_k
    max_k = compute_max_k()
    # Slider の上限を更新（matplotlib >=3.8 は set_valmax、古い版は private属性で代替）
    try:
        s_k.valmax = max_k   # 互換目的：古いバージョンでも動くように
        s_k.ax.set_xlim(s_k.valmin, s_k.valmax)
    except Exception:
        pass
    # 現在の k を補正
    k_now = int(round(s_k.val))
    if k_now > max_k:
        s_k.set_val(max_k)

def on_slide_k(val):
    k = int(round(val))
    if k != int(round(s_k.val)):
        s_k.set_val(k)  # つまみ表示を整数位置に
        return          # set_val が再度この関数を呼ぶので一度返す
    redraw_k(k)

def on_slide_n(val):
    global n_cycles
    n_cycles = int(round(val))
    # n_cycles 変更に伴い k 範囲を更新
    refresh_slider_ranges()
    # 現在の k で再描画
    redraw_k(int(round(s_k.val)))

s_k.on_changed(on_slide_k)
s_n.on_changed(on_slide_n)
# ① ページ送り：n_cycles 山ぶん一気に移動
def step_pages(delta_blocks: int):
    # 現在の k（開始サイクル）
    k_now = int(round(s_k.val))
    # ページ数 × n_cycles だけ進める/戻す
    k_new = k_now + delta_blocks * n_cycles
    # 範囲内にクリップ
    k_new = max(0, min(max_k, k_new))
    # スライダー更新 → on_slide_k 経由で redraw
    s_k.set_val(k_new)

# ② 矢印キーでページ送り（←/↓: 前ページ, →/↑: 次ページ）
def on_key(event):
    if event.key in ("left", "down"):
        step_pages(-1)
    elif event.key in ("right", "up"):
        step_pages(1)

fig.canvas.mpl_connect("key_press_event", on_key)

# ③ マウスホイールでもページ送り（上=前ページ / 下=次ページ）
def on_scroll(event):
    if event.button == "up":
        step_pages(-1)
    elif event.button == "down":
        step_pages(1)

fig.canvas.mpl_connect("scroll_event", on_scroll)


# 初期表示
refresh_slider_ranges()
redraw_k(0)
plt.show()

