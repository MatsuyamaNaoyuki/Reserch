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


def build_mag_sequences(mag9_std,type_end_list, L=16, stride=1):
    """
    typedf: 1 trial の末尾インデックス（昇順）を仮定
    各 trial 内で過去Lフレームを因果窓で切り出し:
      seq_mag: [Nseq, L, 9]
      mu/y は窓の“現在”=末尾フレームに合わせて作るため、その時点の index 配列 js も返す
    """
    type_end_list = type_end_list.tolist()
    type_end_list.insert(0, 0)
    total_span = (L - 1) * stride

    seq_mag, js_all = [], []
    nan_mask = torch.isnan(mag9_std).any(dim=1)


    # rot3_std, y_std は“現在”用のインデックス js で拾うのでここでは触らない

    for i in range(len(type_end_list)-1):

        start = type_end_list[i] + total_span
        end   = type_end_list[i+1]
        if end <= start:
            continue

        js = torch.arange(start, end, device=mag9_std.device)  # 現在時刻の位置
        # 過去Lフレームのインデックスを作る
        rel = torch.arange(L-1, -1, -1, device=mag9_std.device) * stride   
         
        idx = js.unsqueeze(1) - rel    # [num, L]
        # ここで NaN を含む行を除外（必要なら）

        valid_mask = ~nan_mask[idx].any(dim=1)


        idx = idx[valid_mask]
        js = js[valid_mask]


        seq_mag.append(mag9_std[idx])  # [num, L, 9]
        js_all.append(js)

    seq_mag = torch.cat(seq_mag, dim=0) if len(seq_mag)>0 else torch.empty(0, L, 9, device=mag9_std.device)
    js_all  = torch.cat(js_all, dim=0)  if len(js_all)>0  else torch.empty(0, device=mag9_std.device, dtype=torch.long)
    # rot3_std/js_all からMDNを呼び、y_std/js_all を教師に使うため、js_allも返す
    return seq_mag, js_all


def build_selector_batch(mdnn, std_rotate_data, std_mag_data, std_y_data):
    with torch.no_grad():
        pi, mu, sigma = mdnn(std_rotate_data)

    d1 = torch.norm(mu[:, 0, :] - std_y_data, dim=1)
    d2 = torch.norm(mu[:, 1, :] - std_y_data, dim=1)
    label = (d2 < d1).long()
    


    feats = [mu[:, 0, :], mu[:, 1, :], std_mag_data]
    feats = torch.cat(feats, dim=1)

    return feats, label
  

def huber_vec(y_hat, y_gt, delta=1.0):   # [B,3] -> [B]
    r = (y_hat - y_gt).abs()
    quad = 0.5 * (r**2) / delta
    lin  = r - 0.5*delta
    return torch.where(r <= delta, quad, lin).sum(dim=1)

def train(selector, train_loader, optimizer, class_weight, alpha = 0.2, gamma = 5e-5, delta= 1.0):
    selector.train()
    loss_sum_w = 0.0     # Σ (loss_each * w)
    w_sum      = 0.0     # Σ w

    total_correct = 0
    total_count = 0
    delta_sum = 0.0
    delta_count = 0

    for mag_seq,mu_pair, label ,w ,y_std in train_loader:
        mag_seq = mag_seq.to(device, non_blocking=True)          # (B, L, 9)
        mu_pair = mu_pair.to(device, non_blocking=True)          # (B, 6)
        label   = label.to(device, non_blocking=True)            # (B,)
        w       = w.to(device, non_blocking=True)  
        y_std   = y_std.to(device, non_blocking=True)  

        logits, d_n, d_c = selector(mag_seq, mu_pair)

        mu_n, mu_c = mu_pair[:, :3], mu_pair[:, 3:]      # [B,3] / [B,3]
        y_n = mu_n + d_n
        y_c = mu_c + d_c

        sel_c = (label == 1).float().unsqueeze(1)        # [B,1]
        sel_n = 1.0 - sel_c
        y_right = sel_c * y_c + sel_n * y_n              # [B,3] 正解側のみで回帰

        l_reg = huber_vec(y_right, y_std, delta = delta)
        l_cls = F.cross_entropy(logits, label, weight=class_weight, reduction='none', label_smoothing=0.05   )
        L_res = (d_n.pow(2).sum(dim=1) + d_c.pow(2).sum(dim=1))  # [B] Δを小さく

        loss_each = l_reg + alpha * l_cls + gamma * L_res  

        loss = (loss_each * w).sum() / (w.sum() + 1e-8)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta_norm = torch.cat([d_n, d_c], dim=0).norm(dim=1)  # [2B]
            delta_sum += delta_norm.sum().item()
            delta_count += delta_norm.numel()
            loss_sum_w += (loss_each * w).sum().item()
            w_sum      += w.sum().item()
            pred = logits.argmax(dim=1)
            correct = (pred == label).sum().item()
            total_correct += correct

            total_count += mu_pair.size(0)
    ave_delta = delta_sum / max(delta_count, 1)
    ave_loss = loss_sum_w / max(w_sum, 1e-8)  
    ave_correct = total_correct / total_count
    return ave_loss, ave_correct, ave_delta


def test(selector, test_loader,class_weight, alpha = 0.2, gamma = 2e-4, delta= 1.0):
    selector.eval()

    total_loss    = 0.0
    total_correct = 0
    total_count   = 0
    delta_sum     = 0.0
    delta_count   = 0

    with torch.no_grad():
        for mag_seq, mu_pair, label ,y_std in test_loader:
            mag_seq = mag_seq.to(device, non_blocking=True)          # (B, L, 9)
            mu_pair = mu_pair.to(device, non_blocking=True)          # (B, 6)
            label   = label.to(device, non_blocking=True)            # (B,)
            y_std   = y_std.to(device, non_blocking=True)     


            logits, d_n, d_c = selector(mag_seq, mu_pair)
            mu_n, mu_c = mu_pair[:, :3], mu_pair[:, 3:]      # [B,3] / [B,3]
            y_n = mu_n + d_n
            y_c = mu_c + d_c

            sel_c = (label == 1).float().unsqueeze(1)        # [B,1]
            sel_n = 1.0 - sel_c
            y_right = sel_c * y_c + sel_n * y_n              # [B,3] 正解側のみで回帰
            l_reg = huber_vec(y_right, y_std, delta = delta)
            l_cls = F.cross_entropy(logits, label, weight=class_weight, reduction='none', label_smoothing=0.05   )
            L_res = (d_n.pow(2).sum(dim=1) + d_c.pow(2).sum(dim=1))  # [B] Δを小さく

            loss_each = l_reg + alpha * l_cls + gamma * L_res  
            loss = loss_each.mean()

            total_loss   += loss.item() * mu_pair.size(0)
            total_correct += (logits.argmax(dim=1) == label).sum().item()
            total_count   += mu_pair.size(0)

            # Δの大きさも記録（オプション）
            delta_norm = torch.cat([d_n, d_c], dim=0).norm(dim=1)
            delta_sum  += delta_norm.sum().item()
            delta_count += delta_norm.numel()



    ave_loss = total_loss / total_count
    ave_correct = total_correct / total_count
    ave_delta   = delta_sum / max(delta_count, 1)
    return ave_loss, ave_correct, ave_delta

    

def main():
    #変える部分-----------------------------------------------------------------------------------------------------------------
    result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_MDN\selectorres"
    modelpath= r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_MDN\MDN\model.pth"
    filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_MDN\mixhit_fortraintypenan20250715_163007.pickle"
    stride = 1
    L = 16
    kijun = False
    seiki = True
    #-----------------------------------------------------------------------------------------------------------------
    os.makedirs(result_dir, exist_ok=True)
    num_epochs = 500
    batch_size = 128
    r = 0.8
    patience_stop = 30
    patience_scheduler = 10

    

    MDN2= torch.jit.load(modelpath, map_location="cuda:0").eval()


    x_data, y_data, typedf = myfunction.read_pickle_to_torch(filename, motor_angle=True, motor_force=True, magsensor=True)
    y_last3 = y_data[:, -3:]





    if kijun == True:
        base_dir = os.path.dirname(result_dir)
        kijun_dir = myfunction.find_pickle_files("kijun", base_dir)

        kijunx, _ ,_= myfunction.read_pickle_to_torch(kijun_dir,motor_angle=True, motor_force=False, magsensor=False)
        fitA = Mydataset.fit_calibration_torch(kijunx)
        alphaA = torch.ones_like(fitA.amp)
        xA_proc = Mydataset.apply_align_torch(x_data, fitA, alphaA)

        if seiki == True:
            x_max, x_min, x_scale = Mydataset.fit_normalizer_torch(xA_proc)
            std_xdata = Mydataset.apply_normalize_torch(xA_proc, x_min, x_scale)
        else:
            x_mean, x_std = Mydataset.fit_standardizer_torch(xA_proc)
            std_xdata = Mydataset.apply_standardize_torch(xA_proc, x_mean, x_std)
    else:
        if seiki == True:
            x_max, x_min, x_scale = Mydataset.fit_normalizer_torch(x_data)
            std_xdata = Mydataset.apply_normalize_torch(x_data, x_min, x_scale)
        else:
            x_mean, x_std = Mydataset.fit_standardizer_torch(x_data)
            std_xdata = Mydataset.apply_standardize_torch(x_data, x_mean, x_std)


    if seiki == True:
        y_max, y_min, y_scale = Mydataset.fit_normalizer_torch(y_last3)
        std_y_data = Mydataset.apply_normalize_torch(y_last3, y_min,y_scale)
    else:
        y_mean, y_std = Mydataset.fit_standardizer_torch(y_last3)
        std_y_data = Mydataset.apply_standardize_torch(y_last3, y_mean, y_std)



    if seiki == True:
        scaler_data = {
            'x_min': x_min.cpu().numpy(),
            'x_max': x_max.cpu().numpy(),
            'y_min': y_min.cpu().numpy(),
            'y_max': y_max.cpu().numpy(),
        }
    else:
        scaler_data = {
            'x_mean': x_mean.cpu().numpy(),  # GPUからCPUへ移動してnumpy配列へ変換
            'x_std': x_std.cpu().numpy(),
            'y_mean': y_mean.cpu().numpy(),
            'y_std': y_std.cpu().numpy(),
            'fitA': fitA
        }

    scaler_pass = os.path.join(result_dir, "scaler")
    myfunction.wirte_pkl(scaler_data, scaler_pass)



    mask = torch.isfinite(x_data).all(dim=1) & torch.isfinite(y_last3).all(dim=1)

    std_rotate_data, std_force_data, std_mag_data = torch.split(std_xdata, [4, 4, 9], dim=1)
            

    type_end_list = myfunction.get_type_change_end(typedf)
    mag_seq, js = build_mag_sequences(std_mag_data, type_end_list, L=L, stride=stride)




    use_std_rotate = std_rotate_data[js]  # shape: (len(js), ...)
    use_std_y      = std_y_data[js] 

    use_std_rotate = use_std_rotate.to(device)
    use_std_y= use_std_y.to(device)
    mag_seq = mag_seq.to(device)


    with torch.no_grad():
        _, mu_std, _ = MDN2(use_std_rotate)
    
    label, margin, d1, d2 = teacher_label_and_margin(mu_std, use_std_y)
    mu_pair = torch.cat([mu_std[:,0,:], mu_std[:,1,:]], dim=1) 

    #ここまででmu,mag,yの三つがそろったはず
    N = mu_pair.size(0)
    perm = torch.randperm(N)
    n_train = int(N * r)
    idx_tr, idx_te = perm[:n_train], perm[n_train:]



    mu_pair_tr, mu_pair_te = mu_pair[idx_tr], mu_pair[idx_te]
    mag_seq_tr, mag_seq_te = mag_seq[idx_tr], mag_seq[idx_te]
    label_tr, label_te   = label[idx_tr], label[idx_te]
    margin_tr, margin_te = margin[idx_tr], margin[idx_te]
    y_std_tr, y_std_te = use_std_y[idx_tr], use_std_y[idx_te]

    class_weight = get_class_weight(label_tr.to(device))
    class_weight = class_weight.to(device)
    m_w_tr, lo, hi = make_margin_weights(margin_tr.to(device), 0.2, 0.8)
    
    # CEを per-sample で出すため reduction='none'



    train_ds = torch.utils.data.TensorDataset(mag_seq_tr.cpu(),mu_pair_tr.cpu(), label_tr.cpu().long(), m_w_tr.cpu(), y_std_tr.cpu())
    test_ds  = torch.utils.data.TensorDataset(mag_seq_te.cpu(),mu_pair_te.cpu(), label_te.cpu().long(), y_std_te.cpu())
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, pin_memory=True)



    selector = MyModel.SelectorGRUdelta(mag_dim=9, hidden=64, num_layers=1, fc_dim=64).to(device)
    optimizer = torch.optim.Adam(selector.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_scheduler)






    start_epoch = 0
    record_train_loss = []
    record_test_loss = []

    log_dir = os.path.join(result_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    mintestloss = 99999999999
    counter_step = 0
    progress = tqdm(total=num_epochs, initial=start_epoch,desc="Epoch")

    try:
        for epoch in range(num_epochs):
            train_loss, train_correct, train_delta = train(selector, train_loader, optimizer, class_weight)
            test_loss, test_correct , test_delta= test(selector, test_loader, class_weight)
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
                tqdm.write(f"[{epoch}] train={train_loss:.5f} test={test_loss:.5f} train_correct={train_correct:.5f} test_correct={test_correct:.5f} train_delta={train_delta:.5f}test_delta={test_delta:.5f}")

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
