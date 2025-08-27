# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import find_peaks
from myclass import myfunction  # ←既存のユーティリティ

# ====== 設定 ======
DATA_PATH = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\mixhit_fortraintypenan20250715_163007.pickle"  # small の元データ
TYPE_COL_ORIG = "type"                             # 元の列名（small を付ける前）
ROTATE_COL    = "rotate3"  
KEYS          = ["Mc5"]                              # 極小値を取る列
# KEYS          = ["force"]                            # 可視化対象（"sensor", "Mc" などでもOK）
# KEYS          = ["sensor1","sensor2","sensor3"]  
# KEYS          = ["sensor4","sensor5","sensor6"]  
# KEYS          = ["sensor7","sensor8","sensor9"]  
INIT_N_CYCLES = 10                                 # 初期の表示山数(スライダーで変更可)
# ===================

def get_minimum(x, distance=150, prominence=5):
    """極小値インデックスを返す（外れ間隔は後段の揃え方で吸収）"""
    minima, _ = find_peaks(-x, distance=distance, prominence=prominence)
    return np.sort(minima).astype(int)

def align_B_on_A_n_cycles(seriesA, seriesB, a_mins, b_mins, k, n_cycles=2):
    """
    A の極小[k]..[k+n_cycles] 区間を x 軸にして、
    B を山単位で切り出して A の先頭位置に重ねる（不足は NaN、余りは捨てる）。
    """
    if k + n_cycles >= len(a_mins) or k + n_cycles >= len(b_mins):
        raise ValueError("k と n_cycles の組み合わせが極小値配列の長さを超えています。")

    a_start = int(a_mins[k])
    a_end   = int(a_mins[k + n_cycles])
    x_idx   = np.arange(a_start, a_end + 1, dtype=int)

    A_win = seriesA.iloc[a_start:a_end + 1].to_numpy()
    B_on_A_win = np.full_like(A_win, np.nan, dtype=float)

    # 各山ごと（k..k+n_cycles-1）で切り出して、A の先頭(a_start)から詰めていく
    cursor = 0  # A 側の貼り付け先カーソル
    for cyc in range(n_cycles):
        a0, a1 = int(a_mins[k + cyc]), int(a_mins[k + cyc + 1])
        b0, b1 = int(b_mins[k + cyc]), int(b_mins[k + cyc + 1])

        segA = seriesA.iloc[a0:a1].to_numpy()
        segB = seriesB.iloc[b0:b1].to_numpy()

        m = min(len(segA), len(segB))
        if m > 0:
            B_on_A_win[cursor:cursor + m] = segB[:m]
        cursor += len(segA)  # A のその山ぶんだけ進める

    return x_idx, A_win, B_on_A_win

# ===== データ読み込み & 前処理（small を type で 0/1 に分割 → A0/notouch に命名）=====
smalldata = myfunction.load_pickle(DATA_PATH)
if not isinstance(smalldata, pd.DataFrame):
    smalldata = pd.DataFrame(smalldata)

# small 接頭辞を付与（以降は smallXXX で統一）
smalldata.columns = ["small" + c if not str(c).startswith("small") else c for c in smalldata.columns]
TYPE_COL = "small" + TYPE_COL_ORIG
ROTATE_touch = "touch_small" + ROTATE_COL
ROTATE_notouch = "notouch_small" + ROTATE_COL

# type で分割
dftouch = smalldata.loc[smalldata[TYPE_COL] == 0].reset_index(drop=True).add_prefix("touch_")
dfnotouch = smalldata.loc[smalldata[TYPE_COL] == 1].reset_index(drop=True).add_prefix("notouch_")

# 並べて参照できるよう横結合（行数は type ごとに違ってもOK：後で揃えて抜き出す）
df = pd.concat([dftouch, dfnotouch], axis=1)

# 可視化対象列の抽出（touch_* と notouch_* の双方を対象）
cols_all = [c for c in df.columns if any(k.lower() in c.lower() for k in KEYS)]
cols_all.sort()

# touch/notouch の対応ペア（尾が同じものを対応付け）
pairs = []
for c in cols_all:
    if c.startswith("notouch_"):
        tail = c[len("notouch_"):]
        touch_col = "touch_" + tail
        if touch_col in df.columns:
            pairs.append((touch_col, c))

if not pairs:
    raise RuntimeError("対応ペアが見つかりませんでした。KEYS を見直してください。")

# 極小値（touch/notouch それぞれの rotate3 から）
if ROTATE_touch not in df.columns or ROTATE_notouch not in df.columns:
    raise KeyError(f"回転列が見つかりません: {ROTATE_touch}, {ROTATE_notouch}")

a_mins = get_minimum(df[ROTATE_touch].to_numpy())
b_mins = get_minimum(df[ROTATE_notouch].to_numpy())

# ===== 可視化：n 山ぶん + スライダー =====
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.18)

linesA = {}
linesB = {}

# 初期描画（最初の k=0）
n_cycles = INIT_N_CYCLES

def compute_max_k():
    # k..k+n_cycles が A/B の両方で有効になる最大 k
    return max(0, min(len(a_mins), len(b_mins)) - (n_cycles + 1))

max_k = compute_max_k()

# 一度だけ line を作っておき、後は set_data で更新
k0 = 0
x0, _, _ = align_B_on_A_n_cycles(df[pairs[0][0]], df[pairs[0][1]], a_mins, b_mins, k0, n_cycles=n_cycles)
for a_col, b_col in pairs:
    _, Awin, Bwin = align_B_on_A_n_cycles(df[a_col], df[b_col], a_mins, b_mins, k0, n_cycles=n_cycles)
    (lA,) = ax.plot(x0, Awin, label=a_col, alpha=0.9, color="red")
    (lB,) = ax.plot(x0, Bwin, label=b_col, alpha=0.9, color="blue")
    linesA[a_col] = lA
    linesB[b_col] = lB

ax.set_title("A(type=0) vs A(type=1) aligned by minima")
ax.set_xlabel("Row Index (touch axis)")
ax.set_ylabel("Value")
ax.grid(True)
ax.legend(ncol=2)

# y 範囲は全体で固定（動的にしない）
ymin, ymax = np.inf, -np.inf
for a_col, b_col in pairs:
    yA = df[a_col].to_numpy(dtype=float)
    yB = df[b_col].to_numpy(dtype=float)
    for arr in (yA, yB):
        arr = arr[np.isfinite(arr)]
        if arr.size:
            ymin = min(ymin, np.min(arr))
            ymax = max(ymax, np.max(arr))
if not np.isfinite(ymin): ymin, ymax = -1.0, 1.0
pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
ax.set_ylim(ymin - pad, ymax + pad)
# ax.set_ylim(680,820)
# ax.set_ylim(-200,500)

def redraw_k(k:int):
    """k から n_cycles 山ぶんに更新"""
    k = int(np.clip(k, 0, max_k))
    x_idx, _, _ = align_B_on_A_n_cycles(df[pairs[0][0]], df[pairs[0][1]], a_mins, b_mins, k, n_cycles=n_cycles)
    ax.set_xlim(x_idx[0], x_idx[-1])
    for a_col, b_col in pairs:
        x_idx, Awin, Bwin = align_B_on_A_n_cycles(df[a_col], df[b_col], a_mins, b_mins, k, n_cycles=n_cycles)
        linesA[a_col].set_data(x_idx, Awin)
        linesB[b_col].set_data(x_idx, Bwin)

    ax.set_title(
        f"A(type=0) vs A(type=1) aligned by minima | k={k} → k+{n_cycles}={k+n_cycles} "
        f"(touch idx: {int(a_mins[k])}..{int(a_mins[k+n_cycles])})"
    )
    fig.canvas.draw_idle()

# スライダー（k と n_cycles）
ax_slider_k = plt.axes([0.12, 0.08, 0.76, 0.04])
ax_slider_n = plt.axes([0.12, 0.03, 0.76, 0.04])

s_k = Slider(ax_slider_k, "Start cycle k", 0, max_k, valinit=0, valstep=1)
s_n = Slider(ax_slider_n, "n_cycles", 1, 10, valinit=n_cycles, valstep=1)

def refresh_slider_ranges():
    global max_k
    max_k = compute_max_k()
    try:
        s_k.valmax = max_k
        s_k.ax.set_xlim(s_k.valmin, s_k.valmax)
    except Exception:
        pass
    if int(round(s_k.val)) > max_k:
        s_k.set_val(max_k)

def on_slide_k(val):
    k = int(round(val))
    if k != int(round(s_k.val)):
        s_k.set_val(k); return
    redraw_k(k)

def on_slide_n(val):
    global n_cycles
    n_cycles = int(round(val))
    refresh_slider_ranges()
    redraw_k(int(round(s_k.val)))

s_k.on_changed(on_slide_k)
s_n.on_changed(on_slide_n)

# ページ送り：n_cycles 山単位で進む/戻る
def step_pages(delta_blocks:int):
    k_now = int(round(s_k.val))
    k_new = max(0, min(max_k, k_now + delta_blocks * n_cycles))
    s_k.set_val(k_new)  # これで on_slide_k → redraw_k が呼ばれる

def on_key(event):
    if event.key in ("left", "down"):
        step_pages(-1)
    elif event.key in ("right", "up"):
        step_pages(1)

def on_scroll(event):
    if event.button == "up":
        step_pages(-1)
    elif event.button == "down":
        step_pages(1)

fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("scroll_event", on_scroll)

# 初期表示
refresh_slider_ranges()
redraw_k(0)
plt.show()
