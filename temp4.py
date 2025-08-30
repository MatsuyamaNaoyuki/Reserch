import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from myclass import myfunction
from myclass import Mydataset
from torch.utils.tensorboard import SummaryWriter
import os 
from tqdm import tqdm
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_margins(mdnn, t3_std, y_std):
    pi, mu_std, sigma = mdnn(t3_std)
    d1 = torch.norm(mu_std[:,0,:] - y_std, dim=1)
    d2 = torch.norm(mu_std[:,1,:] - y_std, dim=1)
    # margin = |d1 - d2|
    margin = (d1 - d2).abs()
    return margin.cpu().numpy()


modelpath= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\MDN2\model.pth"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit1500kaifortrain.pickle"
result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\selecttest"


x_data, y_data, typedf = myfunction.read_pickle_to_torch(filename, motor_angle=True, motor_force=True, magsensor=True)
y_last3 = y_data[:, -3:]

base_dir = os.path.dirname(result_dir)
kijun_dir = myfunction.find_pickle_files("kijun", base_dir)
kijundata, _ ,_= myfunction.read_pickle_to_torch(kijun_dir,motor_angle=True, motor_force=True, magsensor=True)

std_xdata, xdata_mean, xdata_std ,fitA= Mydataset.align_to_standardize_all(kijundata, x_data)

y_mean, y_std = Mydataset.fit_standardizer_torch(y_last3)
std_y_data = Mydataset.apply_standardize_torch(y_last3, y_mean, y_std)

mask = torch.isfinite(x_data).all(dim=1) & torch.isfinite(y_last3).all(dim=1)
x_data_clean = std_xdata[mask]
std_y_data= std_y_data[mask]

std_rotate_data, std_force_data, std_mag_data = torch.split(x_data_clean, [3, 3, 9], dim=1)
MDN2= torch.jit.load(modelpath, map_location="cuda:0")
std_rotate_data = std_rotate_data.to(device)
std_y_data = std_y_data.to(device)
std_mag_data = std_mag_data.to(device)
margins = get_margins(MDN2, std_rotate_data, std_y_data)

plt.figure(figsize=(8,5))
plt.hist(margins, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel("Margin (|d1 - d2|)")
plt.ylabel("Frequency")
plt.title("Distribution of teacher margin")
plt.grid(True)
plt.show()

print("marginの平均:", margins.mean())
print("marginの中央値:", np.median(margins))
print("marginの最小:", margins.min(), " / 最大:", margins.max())