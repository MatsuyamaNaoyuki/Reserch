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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 




def culc_gosa(prediction, ydata):
    dis_array = np.zeros(4)
    # print(ydata)
    for i in range(4):
        pointpred =np.array([prediction[i * 3],prediction[i * 3 + 1],prediction[i * 3 + 2]])
        pointydata = np.array([ydata[3 * i], ydata[3 * i + 1], ydata[3 * i + 2]])
        distance = np.linalg.norm(pointpred - pointydata)
        dis_array[i] = distance
    return dis_array


def get_min_step(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tag = "loss"
    events = ea.Scalars(tag)
    min_event = min(events, key=lambda e: e.value)
    return min_event.step

def create_sample_indices(episode_ends: np.ndarray, sequence_length: int,
                          condition_horizon: int, subsample_interval: int = 1):
    indices = []
    total_required_past = (condition_horizon - 1) * subsample_interval
    total_required = total_required_past + 1 + sequence_length  # +1 for now

    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        for t in range(start_idx + total_required_past, end_idx - sequence_length):
            # この時刻 t を現在としたときに、過去と未来の情報が取れるかをチェック
            buffer_start = t - total_required_past
            buffer_end = t + sequence_length + 1
            if buffer_end > end_idx:
                continue

            indices.append([buffer_start, buffer_end,0, buffer_end - buffer_start])
    return np.array(indices)


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    condition_horizon=32, subsample_interval=1):
    result = dict()
    for key, input_arr in train_data.items():

        sample = input_arr[buffer_start_idx:buffer_end_idx]  # shape: (327, D)

        if key == 'condition':
            # 10フレームおきに、過去31点＋現在1点の32点を取得
            data = sample[:(condition_horizon - 1) * subsample_interval + 1 : subsample_interval]
            data = np.concatenate([data, sample[[-1]]], axis=0)  # 現在を追加（t）
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
    ndata = np.array(ndata)
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class FingerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, future_estimate_horizon, condition_horizon, now_estimate_horizon, use_data, mode, train_stats = None):
        train_split=0.7
        df_dataset = pd.read_pickle(dataset_path).sort_values('time').reset_index(drop=True)
        input_cols = []
        if use_data["motor_angle"]:
            input_cols += [f'rotate{i}' for i in range(1,5)]          # 4
        if use_data["motor_angle"]:
            input_cols += [f'force{i}' for  i in range(1,5)]          # 4
        if use_data["motor_angle"]:
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
            idx2 = int(N_cls * (train_split + (1-train_split)/2))
            if mode == "train":
                X_split.append(X_cls[:idx1])
                Y_split.append(Y_cls[:idx1])
            elif mode == "val":
                X_split.append(X_cls[idx1:idx2])
                Y_split.append(Y_cls[idx1:idx2])
            elif mode == "test":
                X_split.append(X_cls[:])
                Y_split.append(Y_cls[:])


            # 最後に結合する
        episode_ends = [len(row) for row in X_split]
        episode_ends = np.array(episode_ends).cumsum()
        

        X_data = np.concatenate(X_split, axis=0)
        Y_data = np.concatenate(Y_split, axis=0)
        

         
        train_data = {'action': Y_data,'condition': X_data}
        
       

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(episode_ends = episode_ends,sequence_length=future_estimate_horizon,
                                                condition_horizon = condition_horizon)
        
        # compute statistics and normalized data to [-1,1]
        if mode == "train":
            stats = dict()
            for key, data in train_data.items():
                stats[key] = get_data_stats(data)

            
        else:
            stats = train_stats
        normalized_data = dict()
        for key, data in train_data.items():
            normalized_data[key] = normalize_data(data, stats[key])


        self.indices = indices
        self.stats = stats
        self.normalized_data = normalized_data
        self.future_estimate_horizon = future_estimate_horizon
        self.now_estimate_horizon = now_estimate_horizon
        self.condition_horizon = condition_horizon
 

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(train_data=self.normalized_data, sequence_length=self.condition_horizon,
                                  buffer_start_idx=buffer_start_idx, buffer_end_idx=buffer_end_idx,
                                  )
        # discard unused conditionervations
        nsample['condition'] = nsample['condition'][:self.condition_horizon,:]
        nsample['action'] = nsample['action'][:self.future_estimate_horizon, :]  # ←これが必要！
        
        return nsample
    
def sample_denoised_action(model, condition, noise_scheduler, device, num_diffusion_steps=100):
    model.eval()
    with torch.no_grad():
        if isinstance(condition, np.ndarray):
            condition = torch.tensor(condition, dtype=torch.float32)
        condition = condition.unsqueeze(0)  # (1, 32, 17)
        global_cond = condition.flatten(start_dim=1).to(device)

        # 初期ノイズ: (1, 12, 16)
        x = torch.randn((1, 16, model.final_conv[-1].out_channels), device=device)

        for t in reversed(range(num_diffusion_steps)):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_tensor, global_cond)
            x = noise_scheduler.step(pred_noise, t_tensor, x).prev_sample

        # x: (1, 12, 16) → (16, 12)
        x = x.squeeze(0)

        # 最初の1ステップだけ抽出
        last_step = x[1]  # shape = (12,)
        return last_step.cpu()




def main():

    #変える部分-----------------------------------------------------------------------------------------------------------------

    log_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_Diffusion\all_use3\logs\loss_test"
    filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_Diffusion\mixhit_fortest.pickle"
    usedata = {"motor_angle" : True, "motor_force" : True, 'magsensor' : True}


    #-----------------------------------------------------------------------------------------------------------------
    resultdir = os.path.dirname(os.path.dirname(log_dir))

    num_epochs = 10
    BATCH_SIZE      = 64          # GPU メモリに合わせて調整
    NUM_WORKERS     = min(os.cpu_count(), 8)  # CPU コア数以内
    PIN_MEMORY      = torch.cuda.is_available()
    DROP_LAST       = True  



    future_estimate_horizon = 16
    condition_horizon = 32
    now_estimate_horizon = 1
    condition_dim = 4 * usedata["motor_angle"] + 4 * usedata["motor_force"] + 9 * usedata["magsensor"]
    estimation_dim = 12


    train_stats_dir = myfunction.find_pickle_files("stats", resultdir)
    train_stats = pd.read_pickle(train_stats_dir)

    test_data = FingerDataset(dataset_path = filename, future_estimate_horizon=future_estimate_horizon,
                            condition_horizon=condition_horizon, now_estimate_horizon=1,use_data=usedata, mode = "test", train_stats=train_stats)

    minstep = str(get_min_step(log_dir))
    print(f"使用したephoch:{minstep}")
    modelpath = myfunction.find_pickle_files("epoch" + minstep + "_", directory=resultdir, extension=".pth")
    noise_pred_net = MyModel.ConditionalUnet1D(input_dim=estimation_dim, global_cond_dim=condition_dim*condition_horizon).to(device)

    noise_pred_net.load_state_dict(torch.load(modelpath, weights_only= True))
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_iters,
                                    beta_schedule='squaredcos_cap_v2',clip_sample=True,prediction_type='epsilon')
    for name in dir(noise_scheduler):
        attr = getattr(noise_scheduler, name)
        if isinstance(attr, torch.Tensor):
            setattr(noise_scheduler, name, attr.to(device))
# 他にも scheduler によって必要なテンソルがあるかも（debug print で確認）

    noise_pred_net.eval()

    roundtimes = 1000
    if roundtimes > len(test_data):
        roundtimes = len(test_data)
    dis_array = np.zeros((roundtimes, 4))
    prediction_array = np.zeros((roundtimes, 12))
    real_val_array = np.zeros((roundtimes, 12))
    start= time.time()  # 現在時刻（処理完了後）を取得

    with tqdm(range(roundtimes), desc='Epoch') as tglobal:
        for i in tglobal:
            with torch.no_grad():
                prediction =  sample_denoised_action(noise_pred_net, test_data[i]['condition'],noise_scheduler,device)
            prediction = unnormalize_data(prediction, train_stats['action'])
            prediction_array[i,:] = prediction
            real_val = test_data[i]['action']
            real_val = unnormalize_data(real_val, train_stats['action'])
            real_val = real_val[0]
            real_val_array[i, :] = real_val
            distance = culc_gosa(prediction.tolist(), real_val.tolist())
            dis_array[i, :] = distance

    end = time.time()
    t = (end - start) / roundtimes
    column_means = np.mean(dis_array, axis=0)
    print("列ごとの平均:", column_means.round(2))
    print("かかった時間:", t)
    myfunction.wirte_pkl(prediction_array, "prediction")
    myfunction.wirte_pkl(real_val_array, "real_val")
if __name__ == '__main__':
    main()


 