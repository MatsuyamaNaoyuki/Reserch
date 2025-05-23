from myclass import myfunction



import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pickle
# 例：ダミーデータの作成
filepath = r"C:\Users\shigf\Downloads\mixhit_3000_with_type.pickle"
# df = myfunction.load_pickle(filepath)
with open(filepath, mode='br') as fi:
        data = pickle.load(fi)
        df = data
# フィルタ設定（Butterworthローパス）
fs = 10     # サンプリング周波数（Hz）
cutoff = 0.5    # カットオフ周波数（Hz）
order = 2
b, a = butter(order, cutoff / (fs / 2), btype='low')

# フィルタをかけたい列名
target_columns = ['rotate1','rotate2','rotate3','rotate4',
                  'force1','force2','force3','force4',
                  'sensor1','sensor2','sensor3','sensor4','sensor5',
                  'sensor6', 'sensor7','sensor8','sensor9']

# 各列に対してフィルタ適用
for col in target_columns:
    raw_data = df[col].values
    filtered = filtfilt(b, a, raw_data)
    df[col] = filtered  

myfunction.wirte_pkl(df,"mixhit_3000_with_type_fliter")