import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from myclass import myfunction
import torch
from myclass import Mydataset

# ===== 設定 =====
PATH = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\0818_tubefinger_kijun_rere20250820_173021.pickle"
# KEYS = ["rotate", "force"] 
# KEYS = ["sensor"] 
KEYS = ["Mc"] 
WINDOW = 5000          # 1画面で表示する点数
STEP   = 5000           # ←→・ホイールで動くステップ幅（点）
# =================

xdata, ydata, type= myfunction.read_pickle_to_torch(PATH, True, True, True)

kijunx, kijuny, typedf = myfunction.read_pickle_to_torch(PATH, True, True, True)

fitA = Mydataset.fit_calibration_torch(kijunx)

alphaA = torch.ones_like(fitA.amp)

print(alphaA)
data = Mydataset.fit_calibration_torch(xdata)

print(data)