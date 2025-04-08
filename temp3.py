import matplotlib.pyplot as plt
import numpy as np

# データ
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]  # 平均値
errors = [2, 3, 2.5, 4]    # 標準偏差や誤差範囲

# グラフの描画
plt.figure(figsize=(8, 5))
plt.bar(categories, values, yerr=errors, capsize=10, color='skyblue', edgecolor='black', error_kw={'elinewidth':2, 'ecolor':'black'})

# ラベルの設定
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Chart with Error Bars (Box Plot Style)')

# グリッドの追加
plt.grid(axis='y', linestyle='--', alpha=0.7)

# グラフの表示
plt.show()
