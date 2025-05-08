import numpy as np
import torch
from torch.utils.data import Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from diffusers import UNet1DModel
from diffusers import DDPMScheduler
import os, time
import torch.nn.functional as F

class FingerDataset(Dataset):
    def __init__(self, pickle_path, T=32, motor_angle=True, motor_force=True, magsensor=True,
                 train_split=0.7, mode='train'):
        """
        mode : 'train' | 'val' | 'test'
        T    : ウインドウ長（時系列ステップ数）
        """
        self.T = T
        self.mode = mode

        # 1) --------- DataFrame 読み込み & 列抽出 ------------------
        df = pd.read_pickle(pickle_path).sort_values('time').reset_index(drop=True)

        input_cols = []
        if motor_angle:
            input_cols += [f'rotate{i}' for i in range(1,5)]          # 4
        if motor_force:
            input_cols += [f'force{i}' for  i in range(1,5)]          # 4
        if magsensor:
            input_cols += [f'sensor{i}' for i in range(1,10)]         # 9  (sensor1‑9)

        target_cols = [f'Mc{j}{ax}' for j in range(2,6) for ax in ('x','y','z')]  # 12

        X = df[input_cols].to_numpy(dtype=np.float32)   # (N,17)
        Y = df[target_cols].to_numpy(dtype=np.float32)  # (N,12)

        # 2) --------- データ分割 (時系列順) -------------------------
        X_all = df[input_cols].to_numpy(dtype=np.float32)
        Y_all = df[target_cols].to_numpy(dtype=np.float32)
        labels = df['type'].to_numpy()  # 0,1,... で複数クラス対応

        # 分けたものをここに溜める
        X_split, Y_split = [], []

        # 各クラス（typeごと）に分けて処理
        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]  # このクラスのインデックスを取得
            X_cls = X_all[cls_indices]
            Y_cls = Y_all[cls_indices]
            
            N_cls = len(X_cls)
            idx1 = int(N_cls * train_split)
            idx2 = int(N_cls * (train_split + (1-train_split)/2))
            if self.mode == 'train':
                X_split.append(X_cls[:idx1])
                Y_split.append(Y_cls[:idx1])
            elif self.mode == 'val':
                X_split.append(X_cls[idx1:idx2])
                Y_split.append(Y_cls[idx1:idx2])
            else:  # 'test'
                X_split.append(X_cls[idx2:])
                Y_split.append(Y_cls[idx2:])

            # 最後に結合する
        X = np.concatenate(X_split, axis=0)
        Y = np.concatenate(Y_split, axis=0)


        # 3) --------- チャネルごとの標準化 --------------------------
        self.mean_x, self.std_x = X.mean(0), X.std(0)
        self.mean_y, self.std_y = Y.mean(0), Y.std(0)

        self.xn = (X - self.mean_x) / self.std_x
        self.yn = (Y - self.mean_y) / self.std_y
    


    # 4) ------------------------------------------------------------
    def __len__(self):
        return len(self.xn) - self.T          # ウインドウの数

    def __getitem__(self, idx):
        xs = torch.from_numpy(self.xn[idx:idx+self.T].T)   # (C_in , T)
        ys = torch.from_numpy(self.yn[idx:idx+self.T].T)   # (C_out, T)
        return xs.float(), ys.float()



def one_trin():



def train(model: torch.nn.Module, scheduler, train_loader, n_epoch: int = 200, val_loader=None, grad_accum: int = 1, device: str = "cuda",          # "cpu" でも可
):
    model.train()
    for epoch in range(1, n_epoch + 1):
        t0, losses = time.time(), []
        # ------- 1 エポック学習 -------
        for step, (cond, target) in enumerate(train_loader, 1):
            cond, target = cond.to(device), target.to(device)

            # 1) ランダム時刻をサンプリング
            t = torch.randint(0, scheduler.config.num_train_timesteps,(cond.size(0),), device=device ).long()

            # 2) 前向き拡散でノイズ付与
            noise  = torch.randn_like(target)
            noisy  = scheduler.add_noise(target, noise, t)   # x_t

            # 3) チャネル結合 (early fusion)
            xtotal = torch.cat([noisy, cond], dim=1)         # (B, 12+COND_DIM, T)

            # 4) UNet1DModel でノイズ推定
            pred   = model(xtotal, timesteps=t)

            # 5) ε‑prediction loss
            loss = F.mse_loss(pred, noise, reduction="mean") / grad_accum
            loss.backward()

            # 6) オプティマイザ更新
            if step % grad_accum == 0:
                opt.step(); opt.zero_grad()

            losses.append(loss.item() * grad_accum)

        print(f"[{epoch:03d}/{n_epoch}] "
              f"train_loss={np.mean(losses):.5f}  "
              f"time={time.time()-t0:.1f}s", end="")

        # ------- 検証 (オプション) -------
        if val_loader is not None:
            model.eval(); vlosses = []
            with torch.no_grad():
                for cond, target in val_loader:
                    cond, target = cond.to(device), target.to(device)
                    t      = torch.randint(0, scheduler.config.num_train_timesteps,
                                           (cond.size(0),), device=device).long()
                    noise  = torch.randn_like(target)
                    noisy  = scheduler.add_noise(target, noise, t)
                    xtotal = torch.cat([noisy, cond], dim=1)
                    pred   = model(xtotal, t)
                    vlosses.append(F.mse_loss(pred, noise).item())
            print(f" | val_loss={np.mean(vlosses):.5f}")
            model.train()
        else:
            print()

@torch.no_grad()
def sample(model, cond, steps=100):
    noise_scheduler.set_timesteps(steps)
    x = torch.randn(cond.shape[0], 12, T, device=cond.device)
    for t in noise_scheduler.timesteps:
        model_out = model(x, t, encoder_hidden_states=cond)
        x = noise_scheduler.step(model_out, t, x).prev_sample
    return x          # 標準化空間

#---------------------------------------------------------------------------------- --------------------------------------
motor_angle = True
motor_force = True
magsensor = True
original_result_dir = r"tubefinger_mixhit\GRU_alluse"
data_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger_mixhit\tube_softfinger_mixhit_3000_with_type.pickle"
resume_training = False  # 再開したい場合は True にする


#------------------------------------------------------------------------------------------------------------------------
base_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult"
result_dir = os.path.join(base_path, original_result_dir)
if len(original_result_dir.split(os.sep)) > 1:
    filename = os.path.dirname(result_dir)
    filename = os.path.join(filename, data_name)
else:
    filename = os.path.join(result_dir, data_name)


T = 32
num_epoch = 200
BATCH_SIZE      = 64          # GPU メモリに合わせて調整
NUM_WORKERS     = min(os.cpu_count(), 8)  # CPU コア数以内
PIN_MEMORY      = torch.cuda.is_available()
DROP_LAST       = True  
fingerdata = FingerDataset(pickle_path=data_path, T = T, mode = "train",motor_angle=motor_angle, motor_force= motor_force, magsensor=magsensor)
train_loader = DataLoader(dataset=fingerdata,batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST, persistent_workers=NUM_WORKERS > 0,)
COND_DIM = 4 * motor_angle + 4 * motor_force + 9 * magsensor   # 0〜17
IN_CH    = 12 + COND_DIM


model = UNet1DModel(sample_size=T,in_channels = IN_CH,out_channels=12, block_out_channels=(64,128,256),  layers_per_block=2,)
model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000,beta_schedule="squaredcos_cap_v2")

train(model, noise_scheduler, train_loader)

xtotal = torch.cat([noisy, cond], dim=1)      # (B,29,T)
pred   = unet(xtotal, timesteps=t)




