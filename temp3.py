import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from myclass import myfunction
from myclass import Mydataset
from torch.utils.tensorboard import SummaryWriter
import os 
from tqdm import tqdm

#-----------------------------------------------------------------------------------------------------------------------
result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\temp"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit1500kaifortrain.pickle"
resume_training = False   # 再開したい場合は True にする
kijun = True
seiki = False
#-----------------------------------------------------------------------------------------------------------------------


rotate_data, y_data, typedf = myfunction.read_pickle_to_torch(filename,motor_angle=True, motor_force=False, magsensor=False)
y_last3 = y_data[:, -3:]


if kijun == True:
    base_dir = os.path.dirname(result_dir)
    kijun_dir = myfunction.find_pickle_files("kijun", base_dir)

    kijunx, _ ,_= myfunction.read_pickle_to_torch(kijun_dir,motor_angle=True, motor_force=False, magsensor=False)
    fitA = Mydataset.fit_calibration_torch(kijunx)