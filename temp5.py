import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import os, time, math
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Union
from torch.utils.tensorboard import SummaryWriter
from myclass import myfunction
from myclass import MyModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def create_sample_indices(typedf: np.ndarray, sequence_length: int,
                          condition_horizon: int, data,  subsample_interval: int = 1):
    typedf = typedf.tolist()
    typedf.insert(0, 0)
  
    indices = []
    L=sequence_length + condition_horizon
    total_required_past = (condition_horizon - 1) * subsample_interval
    total_span = total_required_past + 1 + sequence_length  # +1 for now

    nan_mask = np.isnan(data).any(axis=1)
    nan_rows = np.nonzero(nan_mask)[0].tolist()


    nan_rows_set = set(nan_rows)  # 高速化のため set にしておく

    indiceslist = []
    for i in range(len(typedf) - 1):
        start = typedf[i] + total_span
        end = typedf[i + 1]
        if end <= start:
            continue

        js = torch.arange(start, end, device=data.device)
        
        relative_indices = torch.arange(L-1, -1, -1, device=data.device) * subsample_interval
        
        indices = js.unsqueeze(1) - relative_indices  # shape: (num_seq, L)

  
        # # --- ここで NaN 系列を除外する ---
        # indices を CPU に移動して numpy に変換
        indices_np = indices.cpu().numpy()
        # nan_rows が含まれているか判定
        valid_mask = []
        for row in indices_np:
            if any(idx in nan_rows_set for idx in row):
                valid_mask.append(False)  # nan を含む → 無効
            else:
                valid_mask.append(True)   # nan を含まない → 有効

        valid_mask = torch.tensor(valid_mask, device=data.device)

        # 有効な indices だけ残す
        indices = indices[valid_mask]


        if indices.shape[0] == 0:
            continue  # 有効な系列がなければスキップ
        indiceslist.append(indices)



    indiceslist = torch.cat(indiceslist, dim=0)
    result = torch.stack([indiceslist[:, 0], indiceslist[:, -1]], dim=1).numpy()
    print(result)
    print(type(result))
    return result

      

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    condition_horizon=32, subsample_interval=1):
    result = dict()
    for key, input_arr in train_data.items():

        sample = input_arr[buffer_start_idx:buffer_end_idx]  # shape: (327, D)

        if key == 'condition':
            # 10フレームおきに、過去31点＋現在1点の32点を取得
            data = sample[:(condition_horizon) * subsample_interval : subsample_interval]
        elif key == 'action':
            data = sample[-sequence_length:]  # 未来16ステップ（t+1 ~ t+16）
        else:
            raise KeyError("Unknown key in data")

        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }

    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class FingerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, future_estimate_horizon, condition_horizon, now_estimate_horizon, use_data, mode,subsample_interval, traindata = None):
        self.subsample_interval = subsample_interval
        train_split=0.9
        df_dataset = pd.read_pickle(dataset_path)
        input_cols = []
        if use_data["motor_angle"]:
            input_cols += [f'rotate{i}' for i in range(1,5)]          # 4
        if use_data["motor_force"]:
            input_cols += [f'force{i}' for  i in range(1,5)]          # 4
        if use_data["magsensor"]:
            input_cols += [f'sensor{i}' for i in range(1,10)]         # 9  (sensor1‑9)

        target_cols = [f'Mc{j}{ax}' for j in range(2,6) for ax in ('x','y','z')]  # 12

        X_all = df_dataset[input_cols].to_numpy(dtype=np.float32)
        Y_all = df_dataset[target_cols].to_numpy(dtype=np.float32)



        labels = df_dataset['type'].to_numpy()  # 0,1,... で複数クラス対応

        # 分けたものをここに溜める
        X_split, Y_split =  [], []
        


        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]  # このクラスのインデックスを取得
            X_cls = X_all[cls_indices]
            Y_cls = Y_all[cls_indices]

            N_cls = len(X_cls)
            idx1 = int(N_cls * train_split)

            if mode == "train":
                X_split.append(X_cls[:idx1])
                Y_split.append(Y_cls[:idx1])
            elif mode == "val":
                X_split.append(X_cls[idx1:])
                Y_split.append(Y_cls[idx1:])
            elif mode == "test":
                X_split.append(X_cls[:])
                Y_split.append(Y_cls[:])


            # 最後に結合する
        typedf = [len(row) for row in X_split]
        typedf = np.array(typedf).cumsum()
        

        X_data = np.concatenate(X_split, axis=0)
        Y_data = np.concatenate(Y_split, axis=0)
        

         
        train_data = {'action': Y_data,'condition': X_data}
        
       

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(typedf= typedf,sequence_length=future_estimate_horizon,
                                                condition_horizon = condition_horizon, subsample_interval=self.subsample_interval,
                                                data = X_data)



        
        # compute statistics and normalized data to [-1,1]
        if mode == "train":
            stats = dict()
            for key, data in train_data.items():
                stats[key] = get_data_stats(data)

            
        else:
            stats = traindata.stats
        normalized_data = dict()
        for key, data in train_data.items():
            normalized_data[key] = normalize_data(data, stats[key])


        self.indices = indices
        self.stats = stats
        self.normalized_data = normalized_data
        self.future_estimate_horizon = future_estimate_horizon
        self.now_estimate_horizon = now_estimate_horizon
        self.condition_horizon = condition_horizon
        # print(self.normalized_data)
 

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx= self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(train_data=self.normalized_data, sequence_length=self.future_estimate_horizon,
                                  buffer_start_idx=buffer_start_idx, buffer_end_idx=buffer_end_idx,condition_horizon=self.condition_horizon,subsample_interval=self.subsample_interval
                                  )


        # discard unused conditionervations
        # nsample['condition'] = nsample['condition'][:self.condition_horizon,:]
        # nsample['action'] = nsample['action'][:self.future_estimate_horizon, :]  # ←これが必要！
        
        return nsample
    



        


def main():
    #---------------------------------------------------------------------------------- --------------------------------------
    usedata = {"motor_angle" : True, "motor_force" : True, 'magsensor' : True}
    result_dir= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\test"
    # result_dir_name = r"temp"
    data_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\test\testnan20250710_161712.pickle"
    resume_training = False  # 再開したい場合は True にする
    SUBSAMPLE = 1 

    #------------------------------------------------------------------------------------------------------------------------
    # result_dir = os.path.join(os.path.dirname(data_path),result_dir_name)

    num_epochs = 50
    BATCH_SIZE      = 64          # GPU メモリに合わせて調整
    NUM_WORKERS     = min(os.cpu_count(), 8)  # CPU コア数以内
    PIN_MEMORY      = torch.cuda.is_available()
    DROP_LAST       = True  
    


    future_estimate_horizon = 16
    condition_horizon = 32
    now_estimate_horizon = 1
    condition_dim = 4 * usedata["motor_angle"] + 4 * usedata["motor_force"] + 9 * usedata["magsensor"]
    estimation_dim = 12
    train_data = FingerDataset(dataset_path= data_path, future_estimate_horizon=future_estimate_horizon, condition_horizon=condition_horizon, now_estimate_horizon=1,use_data=usedata, mode = "train", subsample_interval=SUBSAMPLE)
    test_data = FingerDataset(dataset_path= data_path, future_estimate_horizon=future_estimate_horizon, condition_horizon=condition_horizon, now_estimate_horizon=1,use_data=usedata, mode = "val", traindata=train_data, subsample_interval=SUBSAMPLE)

    #stasの保存
    stats_pass = os.path.join(result_dir, "stats")
    myfunction.wirte_pkl(train_data.stats, stats_pass)

    
    
    train_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE, shuffle=True, 
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST, persistent_workers=NUM_WORKERS > 0,)
    test_loader = DataLoader(dataset=test_data,batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST, persistent_workers=NUM_WORKERS > 0,)
    # create network object
    noise_pred_net = MyModel.ConditionalUnet1D(input_dim=estimation_dim, global_cond_dim=condition_dim*condition_horizon).to(device)
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_iters,
                                    beta_schedule='squaredcos_cap_v2',clip_sample=True,prediction_type='epsilon')
    ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)







if __name__ == '__main__':
    main()




