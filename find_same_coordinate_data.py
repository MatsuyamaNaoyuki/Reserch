import sys,os, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'myclass'))
import ctypes
import pandas as pd
import numpy as np
def culc_kyori(testrow, trainrow):
    
    testarray = np.array(testrow)
    testarray = testarray[18:]
    testarray = testarray.reshape(4, 3)
    trainarray = np.array(trainrow)
    trainarray = trainarray[18:]
    trainarray = trainarray.reshape(4,3)
    dis_array = np.zeros(4)
    for i in range(4):
        distance = np.linalg.norm(testarray - trainarray)
        dis_array[i] = distance
    average = dis_array.mean()
    return average
    
#     dis_array = np.zeros(4)
#     # print(ydata)
#     for i in range(4):

#         distance = np.linalg.norm(pointpred - pointydata)
#         dis_array[i] = distance
#     return dis_array

testdf = pd.read_pickle(r"C:\Users\shigf\Program\data\sentan_test\modifydata_test20250127_184925.pickle")

traindf = pd.read_csv(r"C:\Users\shigf\Program\data\sentan_morecam\modifydata20250122.csv")


testdf = testdf.drop(testdf.columns[0], axis=1)
traindf = traindf.drop(traindf.columns[0], axis=1)

testdf = (testdf - traindf.mean()) / traindf.std()

traindf = (traindf - traindf.mean()) / traindf.std()


max_row_idx = testdf['rotate1'].idxmax()  # 最大値の行のインデックスを取得
max_row = testdf.loc[max_row_idx]  # その行を取得


near_row = traindf.loc[231081]
# min_ave = float('inf')  # 初期値を無限大に設定
# min_row = None  # 最小値の行を保存する変数

# for _, row in traindf.iterrows():  # DataFrameを1行ずつ取得
#     ave = culc_kyori(max_row, row)
#     if ave < min_ave:  # 最小値を更新
#         min_ave = ave
#         min_row = row

difference_series = max_row - near_row

pd.options.display.float_format = '{:.6f}'.format

        
print(difference_series)
    