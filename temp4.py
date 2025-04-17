import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel2 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
import numpy as np
from nokov.nokovsdk import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


a = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\temp\howtomove20250417_140352.pickle")
print(a)