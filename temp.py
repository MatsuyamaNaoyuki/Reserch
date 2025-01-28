import time
#resnetを実装したもの
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from myclass import myfunction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os, pickle
from myclass import myfunction


def swap_columns(df, col1, col2):
    """
    指定した2つの列を入れ替える関数。

    Parameters:
        df (pd.DataFrame): データフレーム
        col1 (str): 入れ替えたい1つ目の列名
        col2 (str): 入れ替えたい2つ目の列名

    Returns:
        pd.DataFrame: 列を入れ替えた新しいデータフレーム
    """
    # 列の順序をリストとして取得
    columns = list(df.columns)
    
    # 入れ替えたい列のインデックスを取得
    idx1, idx2 = columns.index(col1), columns.index(col2)
    
    # 列を入れ替える
    columns[idx1], columns[idx2] = columns[idx2], columns[idx1]
    
    # 新しい順序でデータフレームを再構築
    return df[columns]



# 指定するディレクトリ
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\test\modifydata_test20250127_184925.pickle"
df = pd.read_pickle(filename)


# columns = df.columns

# print("列名を取得:")
# print(columns)
df = swap_columns(df, 'sensor1', 'sensor3')
df = swap_columns(df, 'sensor4', 'sensor6')
df = swap_columns(df, 'sensor7', 'sensor9')


df.to_pickle('modifydata_test.pickle')