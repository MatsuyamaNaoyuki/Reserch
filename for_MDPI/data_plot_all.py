import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# データ読み込み
df = pd.read_pickle(r"D:\Matsuyama\laerningdataandresult\re3tubefinger0912\mixhit10kaifortest.pickle")
df = pd.DataFrame(df).reset_index()
df = df[:1667]  # 表示対象行数を制限

# 各カテゴリの列

rotate_cols = ["rotate1", "rotate2", "rotate3"]
mag_cols = ["sensor" + str(i) for i in range(1, 10)]
current_cols = ["force1", "force2", "force3"]

# サブプロット作成（縦に3段）
fig, axes = plt.subplots(
    nrows=3, ncols=1, figsize=(12, 6),
    sharex=True,
    gridspec_kw={'hspace': 0.0}  # ← 完全に隙間ゼロに
)

# 1段目：モーター角度
for col in rotate_cols:
    axes[0].plot(df.index, df[col], label=col)
axes[0].set_ylabel("Motor Angle [°]")
axes[0].legend()


# 3段目：電流
for col in current_cols:
    axes[1].plot(df.index, df[col], label=col)
axes[1].set_ylabel("Motor Current [mA]")
axes[1].legend()

# 2段目：磁気センサー
for col in mag_cols:
    axes[2].plot(df.index, df[col], label=col)
axes[2].set_ylabel("Magnetic Sensor[ADC value]")
axes[2].legend(ncol=3)  # 凡例を3列に
axes[2].set_xlabel("Time Step (index)")


for ax in axes:
    ax.grid(True)

plt.tight_layout()
plt.show()