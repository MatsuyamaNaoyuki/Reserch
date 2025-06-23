import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# データ読み込みと整形
df = pd.read_pickle(r"C:\Users\shigf\Program\data\withhit\with_hit_for_test\with_hit_for_test20250226_133722.pickle")
df = pd.DataFrame(df).reset_index()
df = df[:650]

# 各グラフの列と凡例
columns_rotate = ["rotate1", "rotate2", "rotate3", "rotate4"]
columns_current = ["force1", "force2", "force3", "force4"]
columns_mag = ["sensor1", "sensor2", "sensor3", "sensor4", "sensor5", "sensor6", "sensor7", "sensor8", "sensor9"]
label_rotate = ["motor1", "motor2", "motor3", "motor4"]
label_current = ["current1", "current2", "current3", "current4"]
label_mag = ["mag1", "mag2", "mag3", "mag4", "mag5", "mag6", "mag7", "mag8", "mag9"]

# サブプロット（2行1列・x軸共有）
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 5), sharex=True,
                               gridspec_kw={'hspace': 0, 'height_ratios': [1, 1, 1]})  # ←ここで隙間0に

# 上段（モーター角度）
for i, col in enumerate(columns_rotate):
    ax1.plot(df.index, df[col], label=label_rotate[i])
ax1.set_ylabel("Angle [°]")
ax1.legend(loc="upper right")
ax1.grid(True)
ax1.tick_params(labelbottom=False)  # x軸ラベルを非表示（重なり防止）

# 下段（電流）
for i, col in enumerate(columns_current):
    ax2.plot(df.index, df[col], label=label_current[i])
ax2.set_ylabel("Current [mA]")
ax2.set_xlabel("Time step (index)")
ax2.legend(loc="upper right")
ax2.grid(True)

for i, col in enumerate(columns_mag):
    ax3.plot(df.index, df[col], label=label_mag[i])
ax3.set_ylabel("magsensor [V]")
ax3.set_xlabel("Time step (index)")
ax3.legend(loc="upper right")
ax3.grid(True)
plt.tight_layout()
plt.subplots_adjust(hspace=0.0)  # 念のためもう一度隙間をゼロに
plt.show()