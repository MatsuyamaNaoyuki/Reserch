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
import os


result_dir = r"sentan_morecam"
data_dir = r"modifydata20250122.csv"
base_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult"
filename = os.path.join(base_path, result_dir)
if len(result_dir.split(os.sep)) > 1:
    filename = os.path.dirname(filename)


print(filename)