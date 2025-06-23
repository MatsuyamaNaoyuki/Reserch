import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

x = np.linspace(0, 10, 500)
y = ((2 + x) / np.sqrt(1 + (2 + x)**2) - x / np.sqrt(1 + x**2)) * 1250 * 130 / 1000 / 2 + 2.5
y_filtered = np.where(y <= 5.0, y, np.nan)

plt.figure(figsize=(12, 3))
plt.plot(x, y_filtered)

plt.xlabel("Distance between Magnet and Hall Sensor [mm]")
plt.ylabel("Magnetic Sensor Output Voltage [V]")
# plt.title("Magnetic Sensor Output vs Distance")

ax = plt.gca()

# メジャー（主目盛り）設定：軸の数字
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))  # 1 mmごと
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))  # 1 Vごと

# マイナー（補助目盛り）設定：グリッドのみ
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))  # 0.2 mmごとに縦線
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # 0.1 Vごとに横線

# グリッドの描画
ax.grid(True, which='major', linestyle='-', linewidth=0.7)      # 太めの主グリッド（目盛りと一致）
ax.grid(True, which='minor', linestyle='--', linewidth=0.3)     # 細めの補助グリッド
plt.tight_layout()
plt.show()