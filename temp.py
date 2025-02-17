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
import os, sys
from pathlib import Path

path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\currentOK0203\currentOKtest_020320250203_201037.pickle"
data = myfunction.load_pickle(path)
def print_loss_graph(datadf, graphname = None):





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

selected_columns = data[['rotate1', 'rotate2', 'rotate3','rotate4']]

print_loss_graph(selected_columns)