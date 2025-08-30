
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
import math


def build_mag_sequences(mag9_std,typedf, L=16, stride=1):
    """
    typedf: 1 trial の末尾インデックス（昇順）を仮定
    各 trial 内で過去Lフレームを因果窓で切り出し:
      seq_mag: [Nseq, L, 9]
      mu/y は窓の“現在”=末尾フレームに合わせて作るため、その時点の index 配列 js も返す
    """
    typedf = typedf.tolist()
    typedf.insert(0, 0)
    total_span = (L - 1) * stride

    seq_mag, js_all = [], []
    nan_mask = torch.isnan(mag9_std).any(dim=1)
    nan_rows = nan_mask.nonzero(as_tuple=True)[0].tolist()
    nan_rows_set = set(nan_rows) 
    # rot3_std, y_std は“現在”用のインデックス js で拾うのでここでは触らない
    for i in range(len(typedf)-1):

        start = typedf[i] + total_span
        end   = typedf[i+1]
        if end <= start:
            continue

        js = torch.arange(start, end, device=mag9_std.device)  # 現在時刻の位置
        # 過去Lフレームのインデックスを作る
        rel = torch.arange(L-1, -1, -1, device=mag9_std.device) * stride    
        idx = js.unsqueeze(1) - rel    # [num, L]
        # ここで NaN を含む行を除外（必要なら）


        valid_mask = ~nan_mask[idx].any(dim=1)
        myfunction.print_val(valid_mask)


        idx = idx[valid_mask]
        js = js[valid_mask]
        myfunction.print_val(js)
        
        # 今は簡単に実装: そのまま使う
        seq_mag.append(mag9_std[idx])  # [num, L, 9]
        js_all.append(js)

    seq_mag = torch.cat(seq_mag, dim=0) if len(seq_mag)>0 else torch.empty(0, L, 9, device=mag9_std.device)
    js_all  = torch.cat(js_all, dim=0)  if len(js_all)>0  else torch.empty(0, device=mag9_std.device, dtype=torch.long)
    # rot3_std/js_all からMDNを呼び、y_std/js_all を教師に使うため、js_allも返す
    myfunction.print_val(js_all)
    return seq_mag, js_all



mag_std = []

for i in range(101):   # 0～100まで
    row = [i+1000] * 9       # i を8回繰り返す
    mag_std.append(row)


typedf = []
for i in range(50):
    typedf.append(0)

for i in range(49):
    typedf.append(1)

mag_std[1] = [math.nan] * 9 

mag_std = torch.tensor(mag_std)
type_end_list = myfunction.get_type_change_end(typedf)
build_mag_sequences(mag_std,type_end_list, L=16, stride=1)