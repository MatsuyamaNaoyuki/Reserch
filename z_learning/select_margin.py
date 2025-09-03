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


@torch.no_grad()
def make_teacher_and_feats(mdnn, rot3_std_mdn, mag9_std):
    # mdnn: 回転3軸(標準化) -> (pi, mu, sigma)
    pi, mu_std, sigma = mdnn(rot3_std_mdn)          # mu_std: [N, 2, 3]
    # 教師（どっちの mu が GT に近いか）を作るには、別で y_std_mdn が必要
    return pi, mu_std, sigma

@torch.no_grad()
def teacher_label_and_margin(mu_std, y_std):
    # mu_std: [N, 2, 3], y_std: [N, 3]
    d1 = torch.norm(mu_std[:,0,:] - y_std, dim=1)
    d2 = torch.norm(mu_std[:,1,:] - y_std, dim=1)
    teacher = (d2 < d1).long()            # 0 は mu1, 1 は mu2
    margin  = (d1 - d2).abs()             # 近さの差分（自信度 proxy）
    return teacher, margin, d1, d2

def make_margin_weights(margin, low_q = 0.2, high_q = 0.8):
    lo = torch.quantile(margin, low_q)
    hi = torch.quantile(margin, high_q)
    w = (margin - lo) / (hi - lo + 1e-8)
    w = torch.clamp(w, 0.0, 1.0)
    return w, lo, hi


def get_class_weight(teacher):
    # 出現頻度の逆数（簡易）
    num0 = (teacher==0).sum().float()
    num1 = (teacher==1).sum().float()
    w0 = (num0 + num1) / (2.0 * num0 + 1e-8)
    w1 = (num0 + num1) / (2.0 * num1 + 1e-8)
    return torch.tensor([w0, w1], device=teacher.device)




def build_selector_batch(mdnn, std_rotate_data, std_mag_data, std_y_data):
    with torch.no_grad():
        pi, mu, sigma = mdnn(std_rotate_data)

    d1 = torch.norm(mu[:, 0, :] - std_y_data, dim=1)
    d2 = torch.norm(mu[:, 1, :] - std_y_data, dim=1)
    label = (d2 < d1).long()
    


    feats = [mu[:, 0, :], mu[:, 1, :], std_mag_data]
    feats = torch.cat(feats, dim=1)

    return feats, label
  
def train(selector, train_loader, optimizer, class_weight):
    selector.train()

    total_loss = 0
    total_correct = 0
    total_count = 0

    for feats, label ,w in train_loader:
        feats = feats.to(device)
        label = label.to(device)
        w = w.to(device)

        logits, _ = selector(feats)
        loss_each = F.cross_entropy(logits, label, weight=class_weight, reduction='none')
        loss = (loss_each * w).mean()


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


def test(selector, test_loader,class_weight):
    selector.eval()
    
    total_loss = 0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for feats, label in test_loader:
            feats = feats.to(device)
            label = label.to(device)


            logits, _  = selector(feats)
            loss = F.cross_entropy(logits, label, weight=class_weight, reduction='mean')
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
    result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\select_and_margin"
    modelpath= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\MDN2\model.pth"
    filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit1500kaifortrain.pickle"

    #-----------------------------------------------------------------------------------------------------------------

    num_epochs = 500
    batch_size = 128
    r = 0.8
    patience_stop = 30
    patience_scheduler = 10

    selector = MyModel.SelectorNet().to(device)
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


    scaler_data = {
        'x_mean': xdata_mean.cpu().numpy(),  # GPUからCPUへ移動してnumpy配列へ変換
        'x_std': xdata_std.cpu().numpy(),
        'y_mean': y_mean.cpu().numpy(),
        'y_std': y_std.cpu().numpy(),
        'fitA': fitA
    }

    scaler_pass = os.path.join(result_dir, "scaler")
    myfunction.wirte_pkl(scaler_data, scaler_pass)

    with torch.no_grad():
        _, mu_std, _ = MDN2(std_rotate_data)
        label, margin, d1, d2 = teacher_label_and_margin(mu_std, std_y_data)

    feats = torch.cat([mu_std[:,0,:], mu_std[:,1,:], std_mag_data], dim=1)  # [N, 3+3+9]

    N = feats.size(0)
    perm = torch.randperm(N)
    n_train = int(N * r)
    idx_tr, idx_te = perm[:n_train], perm[n_train:]

    feats_tr, feats_te   = feats[idx_tr], feats[idx_te]
    label_tr, label_te   = label[idx_tr], label[idx_te]
    margin_tr, margin_te = margin[idx_tr], margin[idx_te]

    class_weight = get_class_weight(label_tr.to(device))
    m_w_tr, lo, hi = make_margin_weights(margin_tr.to(device), low_q=0.2, high_q=0.8)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='none')

    train_ds = torch.utils.data.TensorDataset(feats_tr.cpu(), label_tr.cpu().long(), m_w_tr.cpu())
    test_ds  = torch.utils.data.TensorDataset(feats_te.cpu(), label_te.cpu().long())
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)











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
            train_loss, train_correct = train(selector, train_loader, optimizer, class_weight)
            test_loss, test_correct = test(selector, test_loader, class_weight)
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
