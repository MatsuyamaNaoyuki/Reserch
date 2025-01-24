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


# 指定するディレクトリ
search_dir = os.path.dirname(os.path.abspath(__file__))

# 'magsensor' を含む .pickle ファイルを検索
def find_pickle_files(directory, keyword):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if keyword in file and file.endswith('.pickle'):
                matched_files.append(os.path.join(root, file))
    return matched_files

# ファイルを検索
pickle_files = find_pickle_files(search_dir, "magsensor")

# 検索結果の確認
if not pickle_files:
    print("該当するファイルが見つかりませんでした。")
else:
    print(f"見つかったファイル: {pickle_files}")
