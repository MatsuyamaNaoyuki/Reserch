
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
from myclass import Mydataset, MyModel
from torch.utils.tensorboard import SummaryWriter
import os 
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   

#損失関数の定義
def mdn_nll(pi, mu, sigma, target):
    B, K, L =mu.shape
    x = target.unsqueeze(1).expand(B,K,L)

    comp_logprob = -0.5 * (((x - mu)/sigma)**2).sum(dim=-1) \
                   - sigma.log().sum(dim=-1) \
                   - 0.5*3*torch.log(torch.tensor(2*3.141592653589793, device=mu.device))
    
    logprob = torch.logsumexp(torch.log(pi + 1e-8) + comp_logprob, dim=1)
    return(-logprob).mean()
    
def train(model, train_loader, optimizer, criterion):
    model.train()
    loss_mean = 0

    for rotate_batch, xyz_batch in train_loader:
        rotate_batch = rotate_batch.to(device).float()
        xyz_batch = xyz_batch.to(device).float()
        pi, mu,sigma = model(rotate_batch)
        loss = criterion(pi, mu, sigma, xyz_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_mean += loss.item() * rotate_batch.size(0)

    loss_mean = loss_mean / len(train_loader.dataset)

    return loss_mean

def test(model, test_loader, criterion):
    model.eval()
    loss_mean = 0
    pi_dev_sum = 0.0
    sigma_mean_sum = 0.0
    sigma_min_sum = 0.0
    n_batches = 0


    with torch.no_grad():
        for rotate_batch, xyz_batch in test_loader:
            rotate_batch = rotate_batch.to(device, non_blocking = True)
            xyz_batch = xyz_batch.to(device, non_blocking = True)
            pi, mu, sigma = model(rotate_batch)
            loss = criterion(pi, mu, sigma, xyz_batch)
            loss_mean += loss.item() * rotate_batch.size(0)

            pi_dev_sum += torch.abs(pi[:, 0] - 0.5).mean().item()
            sigma_mean_sum += sigma.mean().item()
            sigma_min_sum += sigma.min().item()
            n_batches += 1


    avg_pi_dev     = pi_dev_sum / n_batches
    avg_sigma_mean = sigma_mean_sum / n_batches
    avg_sigma_min  = sigma_min_sum / n_batches
    loss_mean = loss_mean / len(test_loader.dataset)

    return loss_mean, avg_pi_dev, avg_sigma_mean, avg_sigma_min






def save_checkpoint(epoch, model, optimizer, record_train_loss, record_test_loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'record_train_loss': record_train_loss,
        'record_test_loss': record_test_loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}.")

def main():
    #-----------------------------------------------------------------------------------------------------------------------
    result_dir = r"D:\Matsuyama\laerningdataandresult\re3tubefingerforMDPI\MDN"
    filename = r"D:\Matsuyama\laerningdataandresult\re3tubefingerforMDPI\mixhit1500kaifortrainbase.pickle"
    resume_training = False   # 再開したい場合は True にする
    kijun = False

    #-----------------------------------------------------------------------------------------------------------------------

    input_dim = 3
    output_class = 2
    output_dim = 12
    num_epochs = 500
    batch_size = 128
    r = 0.8
    patience_stop = 30
    patience_scheduler = 10
    


    model = MyModel.MDN(input_dim=input_dim, output_class=output_class, output_dim=output_dim).to(device)
    criterion = mdn_nll
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_scheduler)

    rotate_data, y_data, typedf = myfunction.read_pickle_to_torch(filename,motor_angle=True, motor_force=False, magsensor=False)
    y_last3 = y_data[:, -1 * output_dim:]


    

    if kijun == True:
        base_dir = os.path.dirname(result_dir)
        kijun_dir = myfunction.find_pickle_files("kijun", base_dir)

        kijunx, _ ,_= myfunction.read_pickle_to_torch(kijun_dir,motor_angle=True, motor_force=False, magsensor=False)
        fitA = Mydataset.fit_calibration_torch(kijunx)
        alphaA = torch.ones_like(fitA.amp)
        rotate_data = Mydataset.apply_align_torch(rotate_data, fitA, alphaA)


    x_min, x_max, y_min, y_max = Mydataset.make_min_max(rotate_data, y_last3, True, False, False,True)

  
    x_scale = (x_max - x_min).clamp(min=1e-8)
    y_scale = (y_max - y_min).clamp(min=1e-8)

    scaler_data = {
        'x_min': x_min.cpu().numpy(),
        'x_max': x_max.cpu().numpy(),
        'y_min': y_min.cpu().numpy(),
        'y_max': y_max.cpu().numpy(),
    }

    rotate_data = (rotate_data - x_min) / x_scale
    y_last3 = (y_last3 - y_min) / y_scale




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





    scaler_pass = os.path.join(result_dir, "scaler")
    myfunction.wirte_pkl(scaler_data, scaler_pass)
    checkpoint_path = os.path.join(result_dir, "3d_checkpoint.pth")



    train_dataset = torch.utils.data.TensorDataset(rotate_data_clean_tr.cpu(),y_last3_clean_tr.cpu())
    test_dataset = torch.utils.data.TensorDataset(rotate_data_clean_te.cpu(),y_last3_clean_te.cpu())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=1, persistent_workers=True)



    record_train_loss = []
    record_test_loss = []

    log_dir = os.path.join(result_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    mintestloss = 99999999999
    counter_step = 0
    progress = tqdm(total=num_epochs, initial=0, desc="Epoch")

    try:
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, criterion)
            test_loss, pi_dev, sigma_mean, sigma_min = test(model, test_loader, criterion)
            record_train_loss.append(train_loss)
            record_test_loss.append(test_loss)
            scheduler.step(test_loss)

            if mintestloss > test_loss:
                counter_step = 0
                filename = os.path.join(result_dir, "model")
                myfunction.save_model(model, filename, time=False)
                mintestloss = test_loss
            else:
                counter_step += 1
                if counter_step >= patience_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break


            if epoch % 10 == 0:
                tqdm.write(f"[{epoch}] train={train_loss:.5f} test={test_loss:.5f} pi_dev={pi_dev:.5f} sigma_mean={sigma_mean:.5f} sigma_min={sigma_min:}")

            writer.add_scalars("loss", {'train':train_loss, 'test':test_loss, 'lr':optimizer.param_groups[0]['lr']}, epoch)
            progress.update(1)     

    except KeyboardInterrupt:
        print("終了します")
        raise
    finally:
        save_checkpoint(epoch, model, optimizer, 
                        record_train_loss, record_test_loss, checkpoint_path)
        myfunction.wirte_pkl(record_test_loss, os.path.join(result_dir, "3d_testloss"))
        myfunction.wirte_pkl(record_train_loss, os.path.join(result_dir, "3d_trainloss"))


        progress.close()
        myfunction.send_message()
if __name__ == '__main__':
    main()
