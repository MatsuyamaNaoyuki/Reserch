import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from myclass import myfunction
# ===== 設定 =====

KEYS = ["force"] 
PATH = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit1500kaifortrain.pickle"
# KEYS = ["rotate"] 
# KEYS = ["sensor1", "sensor2", "sensor3" ] 
# KEYS = ["sensor4", "sensor5", "sensor6" ] 
KEYS = ["sensor7", "sensor8", "sensor9" ] 

# KEYS = ["Mc5"] 
WINDOW = 3000          # 1画面で表示する点数
STEP   = 3000           # ←→・ホイールで動くステップ幅（点）
# =================
 # どれかを含む列を拾う


scaler_data = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\GRUseikika\scaler20250902_181059.pickle")



x_min = (scaler_data['x_min'][0])
x_max = (scaler_data['x_max'][0])
y_min = (scaler_data['y_min'][0])
y_max = (scaler_data['y_max'][0])
x_scale = (x_max - x_min)
y_scale = (y_max - y_min)

retuname = ["rotate1","rotate2", "rotate3", "force1", "force2", "force3", 
            "sensor1", "sensor2", "sensor3", "sensor4", "sensor5", "sensor6", "sensor7", "sensor8", "sensor9",
            "Mc2x", "Mc2y", "Mc2z","Mc3x", "Mc3y", "Mc3z","Mc4x", "Mc4y", "Mc4z","Mc5x", "Mc5y", "Mc5z"]

min_data = pd.Series(np.concatenate([x_min, y_min]),   index=retuname)

scale_data = pd.Series(np.concatenate([x_scale, y_scale]), index=retuname)




# データ読み込み
df = pd.read_pickle(PATH)
df = pd.DataFrame(df).reset_index(drop=True)  # インデックスは0,1,2,...にする



COLUMNS_TO_PLOT = [c for c in df.columns
                   if any(k.lower() in str(c).lower() for k in KEYS)]
COLUMNS_TO_PLOT.sort()

df1 = df.loc[df["type"] == 0, COLUMNS_TO_PLOT + ["type"]].reset_index(drop=True)
df2 = df.loc[df["type"] == 1, COLUMNS_TO_PLOT + ["type"]].reset_index(drop=True)









# 存在チェック
for col in COLUMNS_TO_PLOT:
    if col not in df.columns:
        raise KeyError(f"列が見つかりません: {col}. 利用可能: {list(df.columns)}")

n = len(df)
if n == 0:
    raise ValueError("データが空です。")


# 数値化（文字列が混じっていてもNaNにして続行）
df1[COLUMNS_TO_PLOT] = df1[COLUMNS_TO_PLOT].apply(pd.to_numeric, errors='coerce')

# 列ごとに z-score 標準化: (x - mean) / std
mu1 = df1[COLUMNS_TO_PLOT].mean()
sigma1 = df1[COLUMNS_TO_PLOT].std(ddof=0)        # 母標準偏差（0）で安定化
sigma1 = sigma1.replace(0, np.nan)               # 定数列はゼロ割防止
df_std1 = (df1[COLUMNS_TO_PLOT] - mu1) / sigma1
df_std1 = df_std1.fillna(0)                      # ゼロ割や全NaN区間は0で埋める


df2[COLUMNS_TO_PLOT] = df2[COLUMNS_TO_PLOT].apply(pd.to_numeric, errors='coerce')

# 列ごとに z-score 標準化: (x - mean) / std
mu2 = df2[COLUMNS_TO_PLOT].mean()
sigma2 = df2[COLUMNS_TO_PLOT].std(ddof=0)        # 母標準偏差（0）で安定化
sigma2 = sigma2.replace(0, np.nan)               # 定数列はゼロ割防止
df_std2 = (df2[COLUMNS_TO_PLOT] - mu2) / sigma2
df_std2 = df_std2.fillna(0)     
# 以降、描画はこの SRC を参照（元dfは変更しない）

df1_seiki = (df1[COLUMNS_TO_PLOT] - min_data[COLUMNS_TO_PLOT]) / scale_data[COLUMNS_TO_PLOT] 
df2_seiki = (df2[COLUMNS_TO_PLOT] - min_data[COLUMNS_TO_PLOT]) / scale_data[COLUMNS_TO_PLOT] 
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


datasets = {
    "df1": df1_seiki,
    "df2": df2_seiki,
}

colors = {"df1": "blue", "df2": "red"}  # データごとに色を分ける

lines = {}
for name, data in datasets.items():
    for col in COLUMNS_TO_PLOT:
        y = data[col].iloc[start:end].to_numpy()
        (line,) = ax.plot(
            x, y,
            label=f"{col} ({name})",
            color=colors[name],
            alpha=0.7 if name == "df1" else 0.7,  # 半透明にして重なりを見やすく

        )
        lines[(name, col)] = (line, data)  # line と元データの両方を保持

ax.set_title("Selected Columns (scroll with slider / arrow keys / wheel)")
ax.set_xlabel("Row Index")
ax.set_ylabel("Value")
ax.grid(True)
ax.legend()

def update_ylim(s, e):
    ymin = np.inf
    ymax = -np.inf
    for (name, col), (line, data) in lines.items():
        yy = data[col].iloc[s:e].to_numpy()
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

    for (name, col), (line, data) in lines.items():
        y = data[col].iloc[start_idx:end_idx].to_numpy()
        line.set_data(x, y)

    ax.set_xlim(x[0], x[-1])
    update_ylim(start_idx, end_idx)
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
