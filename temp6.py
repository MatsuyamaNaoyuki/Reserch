import time, os
#resnetを実装したもの
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock
from torch.utils.data import DataLoader, Subset
from myclass import myfunction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import keyboard
from tqdm import tqdm
torch.set_printoptions(threshold=1000)  # デフォルトは1000


def make_sequence_tensor_stride(x, y, typedf,L, stride):
    typedf = typedf.tolist()
    typedf.insert(0, 0)
    total_span = (L - 1) * stride

    seq_x, seq_y = [], []
    
    nan_mask = torch.isnan(x).any(dim=1)
    nan_rows = nan_mask.nonzero(as_tuple=True)[0].tolist()

    nan_rows_set = set(nan_rows)  # 高速化のため set にしておく

    for i in range(len(typedf) - 1):
        start = typedf[i] + total_span
        end = typedf[i + 1]
        if end <= start:
            continue

        js = torch.arange(start, end, device=x.device)
        relative_indices = torch.arange(L-1, -1, -1, device=x.device) * stride
        indices = js.unsqueeze(1) - relative_indices  # shape: (num_seq, L)

        # --- ここで NaN 系列を除外する ---
        # indices を CPU に移動して numpy に変換
        indices_np = indices.cpu().numpy()
        # nan_rows が含まれているか判定
        valid_mask = []
        for row in indices_np:
            if any(idx in nan_rows_set for idx in row):
                valid_mask.append(False)  # nan を含む → 無効
            else:
                valid_mask.append(True)   # nan を含まない → 有効

        valid_mask = torch.tensor(valid_mask, device=x.device)

        # 有効な indices だけ残す
        indices = indices[valid_mask]
        js = js[valid_mask]

        if indices.shape[0] == 0:
            continue  # 有効な系列がなければスキップ
    
        x_seq = x[indices]  # shape: (num_seq_valid, L, D)
        y_seq = y[js]       # shape: (num_seq_valid, D_out)

        seq_x.append(x_seq)
        seq_y.append(y_seq)
            
  
        
    

    seq_x = torch.cat(seq_x, dim=0)
    seq_y = torch.cat(seq_y, dim=0)


    return torch.utils.data.TensorDataset(seq_x, seq_y)



#---------------------------------------------------------------------------------- --------------------------------------
motor_angle = False
motor_force = True
magsensor = True
L = 32 
stride = 10

result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\test"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\test\testnan20250710_161712.pickle"
resume_training = False   # 再開したい場合は True にする


#------------------------------------------------------------------------------------------------------------------------
x_data,y_data, typedf = myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)


x_nan_mask = torch.isnan(x_data).any(dim=1)



mask = ~x_nan_mask
x_data_clean = x_data[mask]
y_data_clean = y_data[mask]

x_mean = x_data_clean.mean(dim=0, keepdim=True)
x_std = x_data_clean.std(dim=0, keepdim=True)
y_mean = y_data_clean.mean(dim=0, keepdim=True)
y_std = y_data_clean.std(dim=0, keepdim=True)


scaler_data = {
    'x_mean': x_mean.cpu().numpy(),  # GPUからCPUへ移動してnumpy配列へ変換
    'x_std': x_std.cpu().numpy(),
    'y_mean': y_mean.cpu().numpy(),
    'y_std': y_std.cpu().numpy()
}
x_data = (x_data - x_mean) / x_std
y_data = (y_data - y_mean) / y_std

scaler_pass = os.path.join(result_dir, "scaler")

type_end_list = myfunction.get_type_change_end(typedf)
seq_dataset = make_sequence_tensor_stride(x_data, y_data,type_end_list, L, stride)



