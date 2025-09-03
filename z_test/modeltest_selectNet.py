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
import japanize_matplotlib
from pathlib import Path
from myclass import Mydataset





def get_uncrrect_num(pred_num):
    if pred_num == 0:
        num = 1
    elif pred_num == 1:
        num = 0

    return num


#変える部分-----------------------------------------------------------------------------------------------------------------

MDNpath= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\MDN_selector\MDNmodel.pth"
selectorpath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\MDN_selector\selector.pth"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\mixhit10kaifortest.pickle"

touch_vis = True
scatter_motor = True
row_data_swith = True
#-----------------------------------------------------------------------------------------------------------------

input_dim = 3

basepath = Path(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult")



x_data, y_data, typedf = myfunction.read_pickle_to_torch(filename, motor_angle=True, motor_force=True, magsensor=True)
y_last3 = y_data[:, -3:]
y_last3 = y_last3.to(device)


mdn_scaler_path = myfunction.find_pickle_files("scaler", os.path.dirname(MDNpath))
mdn_scaler = myfunction.load_pickle(mdn_scaler_path)



select_scaler_path = myfunction.find_pickle_files("scaler", os.path.dirname(selectorpath))
select_scaler_data = myfunction.load_pickle(select_scaler_path)
sel_x_mean = torch.tensor(select_scaler_data['x_mean'],device = device).float()
sel_x_std = torch.tensor(select_scaler_data['x_std'],device = device).float()
sel_y_mean = torch.tensor(select_scaler_data['y_mean'],device = device).float()
sel_y_std = torch.tensor(select_scaler_data['y_std'],device = device).float()
sel_fitA = select_scaler_data['fitA']

mdn_x_mean = torch.tensor(mdn_scaler['x_mean'], device=device).float()
mdn_x_std = torch.tensor(mdn_scaler['x_std'], device=device).float()
mdn_y_mean = torch.tensor(mdn_scaler['y_mean'], device=device).float()
mdn_y_std = torch.tensor(mdn_scaler['y_std'], device=device).float()


alphaA = torch.ones_like(sel_fitA.amp)
xA_proc = Mydataset.apply_align_torch(x_data, sel_fitA, alphaA)
rot3_raw, force3_raw, m9_raw = torch.split(xA_proc, [3,3,9], dim=1)

mdn_x_mean = mdn_x_mean[:3]
mdn_x_std = mdn_x_std[:3]


t3_std_mdn = (rot3_raw.to(device) - mdn_x_mean) / mdn_x_std

m9_session_mean = m9_raw.mean(dim = 0, keepdim=True).to(device)
sel_m9_mean = sel_x_mean[6:15].unsqueeze(0)
m9_raw_shifted = m9_raw.to(device) + (sel_m9_mean - m9_session_mean)


myfunction.print_val(m9_session_mean)
myfunction.print_val(sel_m9_mean)
# m9_std_sel = (m9_raw_shifted - sel_m9_mean) / sel_x_std[6:15].unsqueeze(0)  # (N,9)




# model_MDN = torch.jit.load(MDNpath, map_location="cuda:0")
# model_MDN.eval()
# model_select = torch.jit.load(selectorpath, map_location="cuda:0")
# model_select.eval()



# prediction_array = []


# correct = 0


# with torch.no_grad():
#     for i in range(t3_std_mdn.size(0)):
#     # for i in range(800000, 810000):
#         t3 = t3_std_mdn[i:i+1]
#         m9i = m9_std_sel[i:i+1]
#         pi, mu_std_mdn, sigma = model_MDN(t3)
#         mu_world = mu_std_mdn * mdn_y_std + mdn_y_mean
#         mu_std_sel = (mu_world - sel_y_mean) / sel_y_std
#         feats = torch.cat([mu_std_sel[:,0,:], mu_std_sel[:,1,:], m9i], dim=1)  # (1,15)
#         logits, _ = model_select(feats)      # (1,2)
#         pred_idx = logits.argmax(dim=1).item()
#         un_pred_idx = get_uncrrect_num(pred_idx)
#         # 選ばれたμは“実スケール”で返す（可視化や保存はこちらが正しい）
#         pred_world = mu_world[0, pred_idx, :]            # (3,)
#         unpred_world = mu_world[0, un_pred_idx, :]

#         d_pred = torch.linalg.norm(pred_world - y_last3[i])
#         d_unpred = torch.linalg.norm(unpred_world - y_last3[i])

#         if d_pred.item() < d_unpred.item():
#             correct = correct +1



#         prediction_array.append(pred_world.detach().cpu())

# correct = correct / 10000






# # myfunction.print_val(prediction_array)
# myfunction.print_val(correct)

# parent = os.path.dirname(selectorpath)
# resultpath = os.path.join(parent, "result") 
# myfunction.wirte_pkl(prediction_array, resultpath)

