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

motor_angle = True
motor_force = True
magsensor = False
result_dir = r"sentan_morecam\angle_and_force"
data_name = r"modifydata20250122.csv"
resume_training = False  # 再開したい場合は True にする
csv = True


x_data,y_data = myfunction.read_pickle_to_torch("modifydata20250124_154111.pickle", motor_angle, motor_force, magsensor)

print(f"x_data = {x_data}")
print(f"y_data = {y_data}")
