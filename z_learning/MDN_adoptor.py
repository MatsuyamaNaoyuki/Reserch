
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
import random, math, datetime
import numpy as np
from myclass import MyModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed:int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_loader(filename, result_dir,  kijun):
    r = 0.8
    batch_size = 128


    rotate_data, y_data, typedf = myfunction.read_pickle_to_torch(filename,motor_angle=True, motor_force=False, magsensor=False)
    y_last3 = y_data[:, -3:]


    if kijun == True:
        base_dir = os.path.dirname(result_dir)
        kijun_dir = myfunction.find_pickle_files("kijun", base_dir)

        kijunx, _ ,_= myfunction.read_pickle_to_torch(kijun_dir,motor_angle=True, motor_force=False, magsensor=False)
        fitA = Mydataset.fit_calibration_torch(kijunx)
        alphaA = torch.ones_like(fitA.amp)
        rotate_data = Mydataset.apply_align_torch(rotate_data, fitA, alphaA)

    x_max, x_min, x_scale = Mydataset.fit_normalizer_torch(rotate_data)
    rotate_data = Mydataset.apply_normalize_torch(rotate_data, x_min, x_scale)


    y_max, y_min, y_scale = Mydataset.fit_normalizer_torch(y_last3)
    y_last3 = Mydataset.apply_normalize_torch(y_last3, y_min,y_scale)

    mask = torch.isfinite(rotate_data).all(dim=1) & torch.isfinite(y_last3).all(dim=1)
    rotate_data_clean = rotate_data[mask]
    y_last3_clean     = y_last3[mask]


    N = rotate_data_clean.size(0)
    myfunction.print_val(N)
    perm = torch.randperm(N)
    n_train = int(N * r)
    idx_tr, idx_te = perm[:n_train], perm[n_train:]
    rotate_data_clean_tr, rotate_data_clean_te = rotate_data_clean[idx_tr], rotate_data_clean[idx_te]
    y_last3_clean_tr, y_last3_clean_te =  y_last3_clean[idx_tr], y_last3_clean[idx_te]

    train_dataset = torch.utils.data.TensorDataset(rotate_data_clean_tr.cpu(),y_last3_clean_tr.cpu())
    test_dataset = torch.utils.data.TensorDataset(rotate_data_clean_te.cpu(),y_last3_clean_te.cpu())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=1, persistent_workers=True)

    scaler_data = {
        'x_min': x_min.cpu().numpy(),
        'x_max': x_max.cpu().numpy(),
        'y_min': y_min.cpu().numpy(),
        'y_max': y_max.cpu().numpy(),
        # 'fitA': fitA
    }

    scaler_pass = os.path.join(result_dir, "scaler")
    myfunction.wirte_pkl(scaler_data, scaler_pass)
    return train_loader, test_loader



class MDNwithAdapter(nn.Module):
    def __init__(self, mdn_core:nn.Module, out_dim: int, use_mu_adapter = True, use_pi_tempbias=False):
        super().__init__()
        self.mdn = mdn_core
        self.use_mu_adapter = use_mu_adapter
        self.use_pi_tempbias = use_pi_tempbias

        self.mu_scale = nn.Parameter(torch.ones(out_dim))
        self.mu_bias = nn.Parameter(torch.zeros(out_dim))

        self.log_T = nn.Parameter(torch.zeros(()))
        self.pi_bias = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        pi, mu, sigma = self.mdn(x)
        if self.use_pi_tempbias:
            T = torch.exp(self.log_T) + 1e-6
            pi = pi / T + self.pi_bias
        if self.use_mu_adapter:
            mu = mu * self.mu_scale.view(1,1,-1) + self.mu_bias.view(1,1,-1)
        
        return pi, mu, sigma

def mdn_nll(pi, mu, sigma, target):
    B, K, D =mu.shape

    pi = torch.clamp(pi, 1e-6, 1.0)
    log_pi = torch.log(pi)

    sigma = torch.clamp(sigma, min = 1e-3)

    x = target.unsqueeze(1)
    
    const = -0.5 * D * math.log(2.0 * math.pi)
    log_prob = const \
               - torch.sum(torch.log(sigma), dim=-1) \
               - 0.5 * torch.sum(((x - mu) / sigma) ** 2, dim=-1)  # (B, K)

    log_mix = log_pi + log_prob                    # (B, K)
    logsum = torch.logsumexp(log_mix, dim=1)       # (B,)
    return -logsum.mean()


def train(wrapper, train_loader, optimizer, criterion):
    wrapper.train()
    loss_mean = 0

    for rotate_batch, xyz_batch in train_loader:
        rotate_batch = rotate_batch.to(device).float()
        xyz_batch = xyz_batch.to(device).float()
        pi, mu, sigma = wrapper(rotate_batch)
        loss = criterion(pi, mu, sigma, xyz_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_mean += loss.item() * rotate_batch.size(0)

    loss_mean = loss_mean / len(train_loader.dataset)


    return loss_mean


def test(wrapper, test_loader, criterion):
    wrapper.eval()
    loss_mean = 0




    with torch.no_grad():
        for rotate_batch, xyz_batch in test_loader:
            rotate_batch = rotate_batch.to(device, non_blocking = True)
            xyz_batch = xyz_batch.to(device, non_blocking = True)
            pi, mu, sigma = wrapper(rotate_batch)
            loss = criterion(pi, mu, sigma, xyz_batch)
            loss_mean += loss.item() * rotate_batch.size(0)

  



    loss_mean = loss_mean / len(test_loader.dataset)

    return loss_mean


def save_model(model, filename, time=True):
    if time:
        now = datetime.datetime.now()
        filename =  filename + now.strftime('%Y%m%d_%H%M%S') + '.pth'
    else:
        filename = filename + '.pth'
    model_scripted = torch.jit.script(model)
    model_scripted.save(filename)

def main():
    #-----------------------------------------------------------------------------------------------------------------------
    result_dir = r"D:\Matsuyama\laerningdataandresult\0916another\MDN_adoptor"
    filename = r"D:\Matsuyama\laerningdataandresult\0916another\0916_another_add_10kai.pickle"
    MDNpath = r"D:\Matsuyama\laerningdataandresult\re3tubefinger0912\MDN\model.pth"

    kijun = False
    #-----------------------------------------------------------------------------------------------------------------------


    set_seed(42)
    mdn = torch.jit.load(MDNpath, map_location="cuda:0")
    mdn.eval()

    train_loader, test_loader = make_loader(filename, result_dir, kijun)

    wrapper = MDNwithAdapter(mdn, out_dim=3, use_mu_adapter=True, use_pi_tempbias=False).to(device)
    for p in wrapper.mdn.parameters():
        p.requires_grad = False
    params = [p for p in wrapper.parameters() if p.requires_grad]


    patience_stop = 30
    patience_scheduler = 10
    optimizer = torch.optim.Adam(params, lr=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_scheduler)
    criterion = mdn_nll
    num_epochs = 100

    record_train_loss = []
    record_test_loss = []

    log_dir = os.path.join(result_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    mintestloss = 99999999999
    counter_step = 0
    progress = tqdm(total=num_epochs, initial=0,desc="Epoch")

    try:
        for epoch in range(num_epochs):
            train_loss = train(wrapper, train_loader, optimizer, criterion)
            test_loss = test(wrapper, test_loader, criterion)
            record_train_loss.append(train_loss)
            record_test_loss.append(test_loss)
            scheduler.step(test_loss)


            if mintestloss > test_loss:
                counter_step = 0
                filename = os.path.join(result_dir, "model")
                myfunction.save_model(wrapper, filename, time=False)
                mintestloss = test_loss
            else:
                counter_step += 1
                if counter_step >= patience_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break


            if epoch % 10 == 0:
                tqdm.write(f"[{epoch}] train={train_loss:.5f} test={test_loss:.5f} ")

            writer.add_scalars("loss", {'train':train_loss, 'test':test_loss, 'lr':optimizer.param_groups[0]['lr']}, epoch)
            progress.update(1)     

    except KeyboardInterrupt:
        print("終了します")
        raise
    finally:

        myfunction.wirte_pkl(record_test_loss, os.path.join(result_dir, "3d_testloss"))
        myfunction.wirte_pkl(record_train_loss, os.path.join(result_dir, "3d_trainloss"))


        progress.close()
        myfunction.send_message()



if __name__ == '__main__':
    main()

