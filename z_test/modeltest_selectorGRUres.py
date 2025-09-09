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

MDNpath= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_MDN\MDN\model.pth"
selectorpath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_MDN\selectorres\selector.pth"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_MDN\mixhit_fortesttype.pickle"
L = 16
stride = 1
touch_vis = True
scatter_motor = True
row_data_swith = True
kijun = False
seiki = True
#-----------------------------------------------------------------------------------------------------------------

input_dim = 3





x_data, y_data, typedf = myfunction.read_pickle_to_torch(filename, motor_angle=True, motor_force=True, magsensor=True)
y_last3 = y_data[:, -3:]
y_last3 = y_last3.to(device)


mdn_scaler_path = myfunction.find_pickle_files("scaler", os.path.dirname(MDNpath))
mdn_scaler = myfunction.load_pickle(mdn_scaler_path)



select_scaler_path = myfunction.find_pickle_files("scaler", os.path.dirname(selectorpath))
select_scaler_data = myfunction.load_pickle(select_scaler_path)
if seiki == True:
    sel_x_min = torch.tensor(select_scaler_data['x_min'],device = device).float()
    sel_x_max = torch.tensor(select_scaler_data['x_max'],device=device).float()
    sel_y_min = torch.tensor(select_scaler_data['y_min'],device = device).float()
    sel_y_max = torch.tensor(select_scaler_data['y_max'],device=device).float()
    mdn_x_min = torch.tensor(mdn_scaler['x_min'], device=device).float()
    mdn_x_max = torch.tensor(mdn_scaler['x_max'], device=device).float()
    mdn_y_min = torch.tensor(mdn_scaler['y_min'], device=device).float()
    mdn_y_max = torch.tensor(mdn_scaler['y_max'], device=device).float()
    sel_x_scale = sel_x_max - sel_x_min
    sel_y_scale = sel_y_max - sel_y_min
    mdn_x_scale = mdn_x_max - mdn_x_min
    mdn_y_scale = mdn_y_max - mdn_y_min
    mdn_x_min = mdn_x_min[:4]
    mdn_x_scale = mdn_x_scale[:4] 
    
else:

    sel_x_mean = torch.tensor(select_scaler_data['x_mean'],device = device).float()
    sel_x_std = torch.tensor(select_scaler_data['x_std'],device = device).float()
    sel_y_mean = torch.tensor(select_scaler_data['y_mean'],device = device).float()
    sel_y_std = torch.tensor(select_scaler_data['y_std'],device = device).float()
    mdn_x_mean = torch.tensor(mdn_scaler['x_mean'], device=device).float()
    mdn_x_std = torch.tensor(mdn_scaler['x_std'], device=device).float()
    mdn_y_mean = torch.tensor(mdn_scaler['y_mean'], device=device).float()
    mdn_y_std = torch.tensor(mdn_scaler['y_std'], device=device).float()
    mdn_x_mean = mdn_x_mean[:4]
    mdn_x_std = mdn_x_std[:4]

    if kijun:
        sel_fitA = select_scaler_data['fitA']


if kijun:
    alphaA = torch.ones_like(sel_fitA.amp)
    x_data = Mydataset.apply_align_torch(x_data, sel_fitA, alphaA)

rot3_raw, force3_raw, m9_raw = torch.split(x_data, [4,4,9], dim=1)

if seiki:
    t3_std_mdn = (rot3_raw.to(device) - mdn_x_min) / mdn_x_scale
else:
    t3_std_mdn = (rot3_raw.to(device) - mdn_x_mean) / mdn_x_std

# m9_session_mean = m9_raw.mean(dim = 0, keepdim=True).to(device)
# sel_m9_mean = sel_x_mean[6:15].unsqueeze(0)
# m9_raw_shifted = m9_raw.to(device) + (sel_m9_mean - m9_session_mean)
m9_raw_shifted = m9_raw.to(device)
if seiki:
    m9_std_sel = (m9_raw_shifted - sel_x_min[8:17]) / sel_x_scale[8:17].unsqueeze(0) 
else:
    # m9_std_sel = (m9_raw_shifted - sel_m9_mean) / sel_x_std[6:15].unsqueeze(0) 
    pass
 # (N,9)

type_end_list = myfunction.get_type_change_end(typedf)
mag_seq, js = build_mag_sequences(m9_std_sel, type_end_list, L=L, stride=stride)


use_std_rotate = t3_std_mdn[js]  # shape: (len(js), ...)
use_std_y      = y_last3[js] 


use_std_rotate = use_std_rotate.to(device)
use_std_y= use_std_y.to(device)
mag_seq = mag_seq.to(device)

model_MDN = torch.jit.load(MDNpath, map_location="cuda:0")
model_MDN.eval()
model_select = torch.jit.load(selectorpath, map_location="cuda:0")
model_select.eval()



prediction_array = []


correct = 0


with torch.no_grad():
    for i in range(use_std_rotate.size(0)):
    # for i in range(800000, 810000):
        t3 = use_std_rotate[i:i+1]
        m9i = mag_seq[i:i+1]
        pi, mu_std_mdn, sigma = model_MDN(t3)
        if seiki:
            mu_world = mu_std_mdn * mdn_y_scale + mdn_y_min
            mu_std_sel = (mu_world - sel_y_min) / sel_y_scale
        else:
            mu_world = mu_std_mdn * mdn_y_std + mdn_y_mean
            mu_std_sel = (mu_world - sel_y_mean) / sel_y_std



        mupair = torch.cat([mu_std_sel[:,0,:], mu_std_sel[:,1,:]], dim=1)

        logits, d_n, d_c = model_select(m9i, mupair)   # (1,15)
        mu_n, mu_c = mupair[:, :3], mupair[:, 3:]      # [B,3] / [B,3]
        y_n = mu_n + d_n
        y_c = mu_c + d_c
        
        ypair = torch.stack([y_n, y_c], dim=1)


             # (1,2)
        pred_idx = logits.argmax(dim=1).item()
        un_pred_idx = get_uncrrect_num(pred_idx)
        

        # 選ばれたμは“実スケール”で返す（可視化や保存はこちらが正しい）
        pred_y = ypair[0, pred_idx, :]            # (3,)
        unpred_y = ypair[0, un_pred_idx, :]
        if seiki:
            pred_world = pred_y * mdn_y_scale + mdn_y_min
            unpred_world = unpred_y * mdn_y_scale + mdn_y_min

        else:
            pred_world = pred_y * mdn_y_std + mdn_y_mean
            unpred_world = unpred_y * mdn_y_std + mdn_y_mean
        
        d_pred = torch.linalg.norm(pred_world - use_std_y[i])
        d_unpred = torch.linalg.norm(unpred_world - use_std_y[i])

        if d_pred.item() < d_unpred.item():
            correct = correct +1



        prediction_array.append(pred_world.detach().cpu())

correct = correct / 10000






# myfunction.print_val(prediction_array)
myfunction.print_val(correct)

parent = os.path.dirname(selectorpath)
resultpath = os.path.join(parent, "result") 
myfunction.wirte_pkl(prediction_array, resultpath)
js_path = os.path.join(parent, "js") 
myfunction.wirte_pkl(js, js_path)
