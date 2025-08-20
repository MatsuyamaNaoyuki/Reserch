import time
#resnetを実装したもの


import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os, sys
from pathlib import Path


def get_min_loss_epoch(file_path):
    testdf = pd.read_pickle(file_path)
    testdf = pd.DataFrame(testdf)
    filter_testdf = testdf[testdf.index % 10 == 0]
    minid = filter_testdf.idxmin()
    return minid.iloc[-1]

input_dim = 4
output_dim = 12
learning_rate = 0.001
num_epochs = 300




#変える部分-----------------------------------------------------------------------------------------------------------------

testloss = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\alluse_data32stride1\3d_testloss20250624_192457.pickle"

filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_withhit\withhit_fortest20250227_134803.pickle"



#モデルのロード
minid = str(get_min_loss_epoch(testloss))
print(f"使用したephoch:{minid}")