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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SelectorNet(nn.Module):
    def __init__(self, in_dim = 15, hidden = 128, mode = "cls", use_pi_sigma = False):
        super().__init__()
        self.mode = mode
        self.use_pi_sigma = use_pi_sigma
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
    
        self.head_cls = nn.Linear(hidden, 2)


        self.head_res1 = nn.Linear(hidden, 3)
        self.head_res2 = nn.Linear(hidden, 3)


    def forward(self, feats):
        h = self.shared(feats)
        logits = self.head_cls(h)
        if self.mode == "cls+res":
            d1 = self.head_res1(h)
            d2 = self.head_res2(h)
            return logits, {"d1":d1, "d2":d2}
        else:
            return logits, None
    




def build_selector_batch(mdnn, std_rotate_data, std_mag_data, std_y_data):
    with torch.no_grad():
        pi, mu, sigma = mdnn(std_rotate_data)

    d1 = torch.norm(mu[:, 0, :] - std_y_data, dim=1)
    d2 = torch.norm(mu[:, 1, :] - std_y_data, dim=1)
    label = (d2 < d1).long()


    feats = [mu[:, 0, :], mu[:, 1, :], std_mag_data]
    feats = torch.cat(feats, dim=1)

    return feats, label
  
def train(selector, train_loader, optimizer, criterion):
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


def test(selector, test_loader,criterion):
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

    

def main():
    #変える部分-----------------------------------------------------------------------------------------------------------------
    result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\selecttest"
    modelpath= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\MDN2\model.pth"
    filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortest (1).pickle"

    #-----------------------------------------------------------------------------------------------------------------

    num_epochs = 500
    batch_size = 128
    r = 0.8
    patience_stop = 30
    patience_scheduler = 10

    selector = SelectorNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(selector.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_scheduler)


    MDN2= torch.jit.load(modelpath, map_location="cuda:0")

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
            
    std_rotate_data = std_rotate_data.to(device)
    std_y_data = std_y_data.to(device)
    std_mag_data = std_mag_data.to(device)


    feats, label =build_selector_batch(MDN2, std_rotate_data, std_mag_data, std_y_data)
    feats = feats.cpu()
    label = label.cpu()
    dataset = torch.utils.data.TensorDataset(feats, label)

    scaler_data = {
        'x_mean': xdata_mean.cpu().numpy(),  # GPUからCPUへ移動してnumpy配列へ変換
        'x_std': xdata_std.cpu().numpy(),
        'y_mean': y_mean.cpu().numpy(),
        'y_std': y_std.cpu().numpy(),
        'fitA': fitA
    }

    scaler_pass = os.path.join(result_dir, "scaler")
    myfunction.wirte_pkl(scaler_data, scaler_pass)



    N = len(dataset)
    n_train = int(N * r)
    n_test  = N - n_train 
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test],generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=1, persistent_workers=True)

    start_epoch = 0
    record_train_loss = []
    record_test_loss = []

    log_dir = os.path.join(result_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    mintestloss = 99999999999
    counter_step = 0
    progress = tqdm(total=num_epochs, initial=start_epoch,desc="Epoch")

    try:
        for epoch in range(start_epoch, num_epochs):
            train_loss, train_correct = train(selector, train_loader, optimizer, criterion)
            test_loss, test_correct = test(selector, test_loader, criterion)
            record_train_loss.append(train_loss)
            record_test_loss.append(test_loss)
            scheduler.step(test_loss)

            if mintestloss > test_loss:
                counter_step = 0
                filename = os.path.join(result_dir, "selector")
                myfunction.save_model(selector, filename, time=False)
                mintestloss = test_loss
            else:
                counter_step += 1
                if counter_step >= patience_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break


            if epoch % 10 == 0:
                tqdm.write(f"[{epoch}] train={train_loss:.5f} test={test_loss:.5f} train_correct={train_correct:.5f} test_correct={test_correct:.5f} ")

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
