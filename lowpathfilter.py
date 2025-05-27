from myclass import myfunction



import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pickle

# def  LPF_GC(x,times,sigma):
#     sigma_k = sigma/(times[1]-times[0]) 
#     kernel = np.zeros(int(round(3*sigma_k))*2+1)
#     for i in range(kernel.shape[0]):
#         kernel[i] =  1.0/np.sqrt(2*np.pi)/sigma_k * np.exp((i - round(3*sigma_k))**2/(- 2*sigma_k**2))
        
#     kernel = kernel / kernel.sum()
#     x_long = np.zeros(x.shape[0] + kernel.shape[0])
#     x_long[kernel.shape[0]//2 :-kernel.shape[0]//2] = x
#     x_long[:kernel.shape[0]//2 ] = x[0]
#     x_long[-kernel.shape[0]//2 :] = x[-1]
        
#     x_GC = np.convolve(x_long,kernel,'same')
    
#     return x_GC[kernel.shape[0]//2 :-kernel.shape[0]//2]







# 例：ダミーデータの作成
filepath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\mixhit_fortest20250227_135315.pickle"
# df = myfunction.load_pickle(filepath)
with open(filepath, mode='br') as fi:
        data = pickle.load(fi)
        df = data

df = df.reset_index(drop=True)
# フィルタ設定（Butterworthローパス）
fs = 10     # サンプリング周波数（Hz）
cutoff = 0.5    # カットオフ周波数（Hz）
order = 2
b, a = butter(order, cutoff / (fs / 2), btype='low')

# フィルタをかけたい列名
# target_columns = ['rotate1','rotate2','rotate3','rotate4',
#                   'force1','force2','force3','force4',
#                   'sensor1','sensor2','sensor3','sensor4','sensor5',
#                   'sensor6', 'sensor7','sensor8','sensor9']

target_columns = ['sensor1']

# 各列に対してフィルタ適用
for col in target_columns:
    raw_data = df[col].values
    filtered = filtfilt(b, a, raw_data)
    # df[col] = filtered  
    df[f'{col}_filtered'] = filtered


columns_to_plot = ["sensor1","sensor1_filtered" ]  # 指定したい列のリスト



# df = df[:1000]
# 各列を個別にプロット
plt.figure()  # 新しい図を作成
for column in columns_to_plot:
    plt.plot(df.index, df[column], label=column)  # 列ごとにプロット

plt.title("Selected Columns")
plt.xlabel("Row Index")
plt.ylabel("Value")
plt.legend()  # 凡例を追加
plt.grid(True)
plt.show()
# myfunction.wirte_pkl(df,"mixhit_3000_with_type_fliter")