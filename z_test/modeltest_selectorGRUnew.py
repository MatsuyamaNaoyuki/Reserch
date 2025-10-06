import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
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
from myclass import MyModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def build_mag_sequences(mag9_std,type_end_list, L=16, stride=1):
    """
    typedf: 1 trial の末尾インデックス（昇順）を仮定
    各 trial 内で過去Lフレームを因果窓で切り出し:
      seq_mag: [Nseq, L, 9]
      mu/y は窓の“現在”=末尾フレームに合わせて作るため、その時点の index 配列 js も返す
    """
    type_end_list = type_end_list.tolist()
    type_end_list.insert(0, 0)
    total_span = (L - 1) * stride

    seq_mag, js_all = [], []
    nan_mask = torch.isnan(mag9_std).any(dim=1)
    nan_rows = nan_mask.nonzero(as_tuple=True)[0].tolist()
    nan_rows_set = set(nan_rows) 
    # rot3_std, y_std は“現在”用のインデックス js で拾うのでここでは触らない

    for i in range(len(type_end_list)-1):

        start = type_end_list[i] + total_span
        end   = type_end_list[i+1]
        if end <= start:
            continue

        js = torch.arange(start, end, device=mag9_std.device)  # 現在時刻の位置
        # 過去Lフレームのインデックスを作る
        rel = torch.arange(L-1, -1, -1, device=mag9_std.device) * stride   
         
        idx = js.unsqueeze(1) - rel    # [num, L]
        # ここで NaN を含む行を除外（必要なら）

        valid_mask = ~nan_mask[idx].any(dim=1)


        idx = idx[valid_mask]
        js = js[valid_mask]


        seq_mag.append(mag9_std[idx])  # [num, L, 9]
        js_all.append(js)

    seq_mag = torch.cat(seq_mag, dim=0) if len(seq_mag)>0 else torch.empty(0, L, 9, device=mag9_std.device)
    js_all  = torch.cat(js_all, dim=0)  if len(js_all)>0  else torch.empty(0, device=mag9_std.device, dtype=torch.long)
    # rot3_std/js_all からMDNを呼び、y_std/js_all を教師に使うため、js_allも返す
    return seq_mag, js_all




def get_uncrrect_num(pred_num):
    if pred_num == 0:
        num = 1
    elif pred_num == 1:
        num = 0

    return num


#変える部分-----------------------------------------------------------------------------------------------------------------

MDNpath= r"D:\Matsuyama\laerningdataandresult\re3tubefinger0912\MDN_seikika_hitotu\model.pth"
selectorpath = r"D:\Matsuyama\laerningdataandresult\re3tubefinger0912\selectGRU_category\selector.pth"
filename = r"D:\Matsuyama\laerningdataandresult\re3tubefinger0912\mixhit10kaibase.pickle"
stride = 1
L = 16
magsensor = True
motor_force = False

kijun = False


touch_vis = True
scatter_motor = True
row_data_swith = True
#-----------------------------------------------------------------------------------------------------------------

input_dim = 3
output_class = 2
output_dim = 3





x_data, y_data, typedf = myfunction.read_pickle_to_torch(filename, motor_angle=True, motor_force=True, magsensor=True)
y_last3 = y_data[:, -1 * output_dim:]
y_last3 = y_last3.to(device)

scaler_path = myfunction.find_pickle_files("scaler", os.path.dirname(selectorpath))
scaler_data = myfunction.load_pickle(scaler_path)

x_max = torch.tensor(scaler_data['x_max'],device = device).float()
x_min = torch.tensor(scaler_data['x_min'],device = device).float()
x_scale = x_max - x_min
y_max = torch.tensor(scaler_data['y_max'],device = device).float()
y_min = torch.tensor(scaler_data['y_min'],device = device).float()
y_scale = y_max - y_min


if kijun: 
    sel_fitA = scaler_data['fitA']
    alphaA = torch.ones_like(sel_fitA.amp)
    x_data = Mydataset.apply_align_torch(x_data, sel_fitA, alphaA)

x_data_std = (x_data.to(device) - x_min) / x_scale
y_data_std = (y_last3.to(device) - y_min) / y_scale
rot3_std, force3_std, m9_std = torch.split(x_data_std, [3,3,9], dim=1)

selector_input_data = []

if motor_force:
    selector_input_data.append(force3_std)
if magsensor:
    selector_input_data.append(m9_std)

selector_input_data  = torch.cat(selector_input_data, dim = 1)

# m9_std_sel = (m9_raw_shifted - sel_m9_mean) / sel_x_std[6:15].unsqueeze(0)  # (N,9)

type_end_list = myfunction.get_type_change_end(typedf)
selector_input_seq, js = build_mag_sequences(selector_input_data, type_end_list, L=L, stride=stride)


use_std_rotate = rot3_std[js]  # shape: (len(js), ...)
use_std_y      = y_data_std[js] 


use_std_rotate = use_std_rotate.to(device)
use_std_y= use_std_y.to(device)
selector_input_seq = selector_input_seq.to(device)

model_MDN = torch.jit.load(MDNpath, map_location="cuda:0")
model_MDN.eval()
model_select = torch.jit.load(selectorpath, map_location="cuda:0")
model_select.eval()



prediction_array = []


dis = 0


with torch.no_grad():
    for i in range(use_std_rotate.size(0)):
        t3 = use_std_rotate[i:i+1]
        m9i = selector_input_seq[i:i+1]
        pi, mu_std, sigma = model_MDN(t3)
        mu_world = mu_std * y_scale + y_min

        mu_pair = mu_std.reshape(mu_std.shape[0], -1)  # (1,15)
        logits = model_select(m9i, mu_pair)      # (1,2)
        pred_idx = logits.argmax(dim=1).item()
        pred_world = mu_world[0, pred_idx, :]            # (3,)

        y = use_std_y[i] * y_scale + y_min
        d_pred = torch.linalg.norm(pred_world - y)


        dis = dis + d_pred
        prediction_array.append(pred_world.detach().cpu())






dis = dis / use_std_rotate.size(0)

# myfunction.print_val(prediction_array)
myfunction.print_val(dis)

# parent = os.path.dirname(selectorpath)
# resultpath = os.path.join(parent, "result") 
# myfunction.wirte_pkl(prediction_array, resultpath)
# js_path = os.path.join(parent, "js") 
# myfunction.wirte_pkl(js, js_path)
