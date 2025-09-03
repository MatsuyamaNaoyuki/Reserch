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


#損失関数の定義
    
def MDNtrain(model, train_loader, optimizer, criterion):
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

def MDNtest(model, test_loader, criterion):
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


  
def selectortrain(selector, train_loader, optimizer, criterion):
    selector.train()

    total_loss = 0
    total_correct = 0
    total_count = 0

    for feats, label in train_loader:
        feats = feats.to(device)
        label = label.to(device)
        logits, _ = selector(feats)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total_loss += loss.item() * feats.size(0)
            total_count += feats.size(0)
    
    ave_loss = total_loss / total_count
    ave_correct = total_correct / total_count
    return ave_loss, ave_correct


def selectortest(selector, test_loader,criterion):
    selector.eval()
    
    total_loss = 0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for feats, label in test_loader:
            feats = feats.to(device)
            label = label.to(device)

            logits, _  = selector(feats)
            loss = criterion(logits, label.to(device))

            pred = logits.argmax(dim = 1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total_loss += loss.item() * feats.size(0)
            total_count += feats.size(0)

    ave_loss = total_loss / total_count
    ave_correct = total_correct / total_count
    return ave_loss, ave_correct

def build_selector_batch(mdnn, std_rotate_data, std_mag_data, std_y_data):

    std_rotate_data = std_rotate_data.to(device)
    std_mag_data    = std_mag_data.to(device)
    std_y_data      = std_y_data.to(device)
    mdnn.eval()  # 特徴量抽出だけなら推奨
    with torch.no_grad():
        pi, mu, sigma = mdnn(std_rotate_data)

    d1 = torch.norm(mu[:, 0, :] - std_y_data, dim=1)
    d2 = torch.norm(mu[:, 1, :] - std_y_data, dim=1)
    label = (d2 < d1).long()


    feats = [mu[:, 0, :], mu[:, 1, :], std_mag_data]
    feats = torch.cat(feats, dim=1)

    return feats, label


def main():
    #-----------------------------------------------------------------------------------------------------------------------
    result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\MDN_selector"
    filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\mixhit1500kaifortrain.pickle"

    #-----------------------------------------------------------------------------------------------------------------------

    num_epochs = 500
    batch_size = 128
    r = 0.8
    patience_stop = 30
    patience_scheduler = 10

    MDNmodel = MyModel.MDN2().to(device)
    MDNcriterion = MyModel.mdn_nll
    MDNoptimizer = torch.optim.Adam(MDNmodel.parameters(), lr=1e-3)
    MDNscheduler = lr_scheduler.ReduceLROnPlateau(MDNoptimizer, mode='min', factor=0.5, patience=patience_scheduler)

    selectormodel = MyModel.SelectorNet().to(device)
    selectorcriterion = nn.CrossEntropyLoss()
    selectoroptimizer = torch.optim.Adam(selectormodel.parameters(), lr=1e-3, weight_decay=1e-4)
    selectorscheduler = lr_scheduler.ReduceLROnPlateau(selectoroptimizer, mode='min', factor=0.5, patience=patience_scheduler)



    x_data, y_data, typedf = myfunction.read_pickle_to_torch(filename, motor_angle=True, motor_force=True, magsensor=True)
    y_last3 = y_data[:, -3:]



    base_dir = os.path.dirname(result_dir)
    kijun_dir = myfunction.find_pickle_files("kijun", base_dir)
    kijundata, _ ,_= myfunction.read_pickle_to_torch(kijun_dir,motor_angle=True, motor_force=True, magsensor=True)
    std_xdata, xdata_mean, xdata_std ,fitA= Mydataset.align_to_standardize_all(kijundata, x_data)



    y_mean, y_std = Mydataset.fit_standardizer_torch(y_last3)
    std_y_data = Mydataset.apply_standardize_torch(y_last3, y_mean, y_std)

    mask = torch.isfinite(x_data).all(dim=1) & torch.isfinite(y_last3).all(dim=1)
    x_data_clean = std_xdata[mask]
    std_y_data= std_y_data[mask]

    std_rotate_data, std_force_data, std_mag_data = torch.split(x_data_clean, [3, 3, 9], dim=1)
          

    MDNdataset = torch.utils.data.TensorDataset(std_rotate_data, std_y_data)

    scaler_data = {
        'x_mean': xdata_mean.cpu().numpy(),  # GPUからCPUへ移動してnumpy配列へ変換
        'x_std': xdata_std.cpu().numpy(),
        'y_mean': y_mean.cpu().numpy(),
        'y_std': y_std.cpu().numpy(),
        'fitA': fitA
    }

    scaler_pass = os.path.join(result_dir, "scaler")
    myfunction.wirte_pkl(scaler_data, scaler_pass)


    MDNdataset = torch.utils.data.TensorDataset(std_rotate_data, std_y_data)

    N = len(MDNdataset)
    n_train = int(N * r)
    n_test  = N - n_train 
    MDNtrain_dataset, MDNtest_dataset = torch.utils.data.random_split(MDNdataset, [n_train, n_test],generator=torch.Generator().manual_seed(0))
    MDNtrain_loader = DataLoader(MDNtrain_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True)
    MDNtest_loader = DataLoader(MDNtest_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=1, persistent_workers=True)

    start_epoch = 0
    record_MDN_train_loss = []
    record_MDN_test_loss = []

    log_dir = os.path.join(result_dir, "MDNlogs")
    writer = SummaryWriter(log_dir=log_dir)

    mintestloss = 99999999999
    counter_step = 0
    progress = tqdm(total=num_epochs, initial=start_epoch,desc="Epoch")

    try:
        for epoch in range(num_epochs):
            train_loss = MDNtrain(MDNmodel, MDNtrain_loader, MDNoptimizer, MDNcriterion)
            test_loss, pi_dev, sigma_mean, sigma_min = MDNtest(MDNmodel, MDNtest_loader, MDNcriterion)
            record_MDN_train_loss.append(train_loss)
            record_MDN_test_loss.append(test_loss)
            MDNscheduler.step(test_loss)

            if mintestloss > test_loss:
                counter_step = 0
                filename = os.path.join(result_dir, "MDNmodel")
                myfunction.save_model(MDNmodel, filename, time=False)
                mintestloss = test_loss
            else:
                counter_step += 1
                if counter_step >= patience_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break


            if epoch % 10 == 0:
                tqdm.write(f"[{epoch}] train={train_loss:.5f} test={test_loss:.5f} pi_dev={pi_dev:.5f} sigma_mean={sigma_mean:.5f} sigma_min={sigma_min:}")
            writer.add_scalars("loss", {'train':train_loss, 'test':test_loss, 'lr':MDNoptimizer.param_groups[0]['lr']}, epoch)
            progress.update(1)     

    except KeyboardInterrupt:
        print("終了します")
        raise
    finally:
        myfunction.wirte_pkl(record_MDN_test_loss, os.path.join(result_dir, "MDN_testloss"))
        myfunction.wirte_pkl(record_MDN_train_loss, os.path.join(result_dir, "MDN_trainloss"))
        progress.close()
        myfunction.send_message("MDN終了")

    feats, label =build_selector_batch(MDNmodel, std_rotate_data, std_mag_data, std_y_data)
    feats = feats.cpu()
    label = label.cpu()
    selectordataset = torch.utils.data.TensorDataset(feats, label)

    
    N = len(selectordataset)
    n_train = int(N * r)
    n_test  = N - n_train 
    selectortrain_dataset, selectortest_dataset = torch.utils.data.random_split(selectordataset, [n_train, n_test],generator=torch.Generator().manual_seed(0))
    selectortrain_loader = DataLoader(selectortrain_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True)
    selectortest_loader = DataLoader(selectortest_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=1, persistent_workers=True)


    record_selector_train_loss = []
    record_selector_test_loss = []

    log_dir = os.path.join(result_dir, "selectorlogs")
    writer = SummaryWriter(log_dir=log_dir)

    mintestloss = 99999999999
    counter_step = 0
    progress = tqdm(total=num_epochs, initial=start_epoch,desc="Epoch")

    try:
        for epoch in range(num_epochs):
            train_loss, train_correct = selectortrain(selectormodel, selectortrain_loader, selectoroptimizer, selectorcriterion)
            test_loss, test_correct = selectortest(selectormodel, selectortest_loader, selectorcriterion)
            record_selector_train_loss.append(train_loss)
            record_selector_test_loss.append(test_loss)
            selectorscheduler.step(test_loss)

            if mintestloss > test_loss:
                counter_step = 0
                filename = os.path.join(result_dir, "selector")
                myfunction.save_model(selectormodel, filename, time=False)
                mintestloss = test_loss
            else:
                counter_step += 1
                if counter_step >= patience_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break


            if epoch % 10 == 0:
                tqdm.write(f"[{epoch}] train={train_loss:.5f} test={test_loss:.5f} train_correct={train_correct:.5f} test_correct={test_correct:.5f} ")

            writer.add_scalars("loss", {'train':train_loss, 'test':test_loss, 'lr':selectoroptimizer.param_groups[0]['lr']}, epoch)
            progress.update(1)     

    except KeyboardInterrupt:
        print("終了します")
        raise
    finally:

        myfunction.wirte_pkl(record_selector_test_loss, os.path.join(result_dir, "selector_testloss"))
        myfunction.wirte_pkl(record_selector_train_loss, os.path.join(result_dir, "selector_trainloss"))


        progress.close()
        myfunction.send_message()

if __name__ == '__main__':
    main()
 