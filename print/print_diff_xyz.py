import pickle 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 偏自己相関を計算 & プロットする関数
def plot_pacf(data, lags=100):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    for i, col in enumerate(['Mc5x', 'Mc5y', 'Mc5z']):
        sm.graphics.tsa.plot_pacf(data[col].dropna(), lags=lags, ax=axes[i], method='ywmle')
        axes[i].set_title(f'Partial Autocorrelation of {col}')

    plt.tight_layout()
    plt.show()

# 偏自己相関をプロット


def print_loss_graph(datadf, graphname):





    fig, ax = plt.subplots(figsize = (8.0, 6.0)) 
    datadf.plot(ax=ax, )  # ylimを直接指定
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()

    ax.set_title(graphname,fontsize=20)

    ax.set_xlabel("frame",fontsize=20)
    ax.set_ylabel("diff",fontsize=20)
    ax.set_xticklabels(xticklabels,fontsize=12)
    ax.set_yticklabels(yticklabels,fontsize=12)   
    plt.show()

file_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\moredataset1000\output.pickle"
data = pd.read_pickle(file_path)
output = data[['Mc2x', 'Mc2y', 'Mc2z','Mc3x', 'Mc3y', 'Mc3z','Mc4x', 'Mc4y', 'Mc4z','Mc5x', 'Mc5y', 'Mc5z']]


file_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\currentOK0203\currentOKtest_020320250203_201037.pickle"
data = pd.read_pickle(file_path)
base = data[['Mc2x', 'Mc2y', 'Mc2z','Mc3x', 'Mc3y', 'Mc3z','Mc4x', 'Mc4y', 'Mc4z','Mc5x', 'Mc5y', 'Mc5z']]

diff = output - base 
xyz = diff.loc[:, ['Mc5x', 'Mc5y', 'Mc5z']]
plot_pacf(xyz, lags=4000)

# print_loss_graph(xyz, "error_each_axis")
# autocorr_values = xyz.apply(lambda x: [x.autocorr(lag) for lag in range(0,8000, 100)], axis=0)
# autocorr_df = pd.DataFrame(autocorr_values, index=range(100))  # 0〜10ラグの自己相関

print("自己相関係数:")
# print(autocorr_df)

# # 自己相関プロット
# fig, axes = plt.subplots(3, 1, figsize=(8, 8))
# # 
# for i, col in enumerate(['Mc5x', 'Mc5y', 'Mc5z']):
#     sm.graphics.tsa.plot_acf(xyz[col].dropna(), lags=8000, ax=axes[i])
#     axes[i].set_title(f'Autocorrelation of {col}')



# plt.tight_layout()
# plt.show()

# for col in ['Mc5y', 'Mc5z']:
#     autocorr_vals = [xyz[col].autocorr(lag) for lag in range(1, 5000)]
#     peaks, _ = find_peaks(autocorr_vals, height=0.1)  # ピークを検出

#     print(f"{col} の周期の候補（ラグ）: {peaks}")