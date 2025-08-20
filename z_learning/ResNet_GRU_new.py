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
from myclass.MyModel import ResNetGRU
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import subprocess


#1エポックの学習
def train(model, data_loader, optimizer, criterion):
    model.train()
    loss_mean = 0


    # with profiler.profile(record_shapes=False, use_cuda=True) as prof:
    for j, (x_train_data, y_train_data) in enumerate(data_loader):

        x_train_data = x_train_data.to(device, non_blocking=True)
        y_train_data = y_train_data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x_train_data)
        loss = criterion(y_pred, y_train_data)
        loss.backward()
        optimizer.step()
 

        loss_mean += loss.item()

    loss_mean = loss_mean / (j + 1)
    return loss_mean

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

def test(model, data_loader, criterion):
    # モデルを評価モードにする
    model.eval()
    loss_mean = 0

    for j, (x_train_data, y_train_data) in enumerate(data_loader):
        x_train_data = x_train_data.to(device, non_blocking=True)
        y_train_data = y_train_data.to(device, non_blocking=True)

        y_pred = model(x_train_data)
        loss = criterion(y_pred, y_train_data)
        
        y_pred = model(x_train_data)  # 順伝播（=予測）
        loss = criterion(y_pred, y_train_data)  # 損失を計算
        loss_mean += loss.item()

    loss_mean = loss_mean / (j+1)

    return loss_mean

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

# def load_checkpoint(filepath, model, optimizer):
#     checkpoint = torch.load(filepath)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     return checkpoint['epoch'], checkpoint['record_train_loss'], checkpoint['record_test_loss']

def save_test(test_dataset, result_dir):


    test_indices = test_dataset.indices  # 添え字のみ取得
# **保存パス**
    test_indices_path = os.path.join(result_dir, "test_indices")

    # **pickle ファイルとして保存**
    myfunction.wirte_pkl(test_indices, test_indices_path)


def main():
#---------------------------------------------------------------------------------- --------------------------------------
    motor_angle = True
    motor_force = True
    magsensor = True
    L = 32 
    stride = 20

    result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\nohit_alluse_stride20"
    filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\0816_tubefinger_nohit_1500kai_re20250816_133941.pickle"
    resume_training = False   # 再開したい場合は True にする


    #------------------------------------------------------------------------------------------------------------------------

    
    

    
    # ハイパーパラメータ
    input_dim = 3 * motor_angle + 3 * motor_force + 9 * magsensor
    output_dim = 12
    learning_rate = 0.001
    num_epochs = 500
    batch_size = 128
    r = 0.8
    patience_stop =30
    patience_scheduler = 10

    # モデルの初期化
    model = ResNetGRU(input_dim=input_dim,output_dim=output_dim, hidden=128)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_scheduler)


    x_data,y_data, typedf = myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
    if typedf is None:
        typedf = pd.Series([0] * x_data.shape[0])

    


    print(y_data[0])


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

    print(len(seq_dataset))
    checkpoint_path = os.path.join(result_dir, "3d_checkpoint.pth")



    if resume_training and os.path.exists(checkpoint_path):
        test_indices_path = myfunction.find_pickle_files("test_indices", result_dir)
        test_indices = myfunction.load_pickle(test_indices_path)
        all_indices = set(range(len(seq_dataset)))
        test_indices = set(test_indices)  # Set に変換（高速化）
        train_indices = list(all_indices - test_indices)  # 差分を取る
        train_dataset = Subset(seq_dataset, train_indices)
        test_dataset = Subset(seq_dataset, list(test_indices))  
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True, num_workers=0, persistent_workers=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=0, persistent_workers=False)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        record_train_loss = checkpoint['record_train_loss']
        record_test_loss = checkpoint['record_test_loss']
        print(f"Resuming training from epoch {start_epoch}.")
    else:
        train_dataset, test_dataset = torch.utils.data.random_split(seq_dataset, [r,1-r],generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=1, persistent_workers=True)
        myfunction.wirte_pkl(scaler_data, scaler_pass)
        save_test(test_dataset,result_dir)
        start_epoch = 0
        record_train_loss = []
        record_test_loss = []


    log_dir = os.path.join(result_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)


    mintestloss = 999999999
    counter_stop = 0
    progress = tqdm(total=num_epochs,initial=start_epoch,desc ="Epoch")

    try:
        for epoch in range(start_epoch, num_epochs):
            train_loss = train(model, train_loader, optimizer, criterion)
            test_loss  = test(model , test_loader, criterion)
            record_train_loss.append(train_loss)
            record_test_loss.append(test_loss)
            scheduler.step(test_loss)

            # 10 エポックごとにログを出す
            if mintestloss > test_loss:
                counter_stop = 0
                filename = os.path.join(result_dir, "model")
                myfunction.save_model(model, filename,time=False)
                mintestloss = test_loss
            else:
                counter_stop +=1
                if counter_stop >= patience_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                # filename = '\\3d_model_epoch' + str(epoch)+"_"
                # filename = result_dir + filename
                # myfunction.save_model(model, filename)
                tqdm.write(f"[{epoch}] train={train_loss:.5f} test={test_loss:.5f}")

            writer.add_scalars("loss", {'train':train_loss, 'test':test_loss, 'lr':optimizer.param_groups[0]['lr']}, epoch)
            progress.update(1)           # ★ これでバーが 1 目盛り進む
    except KeyboardInterrupt:
        print("\n[INFO] Detected Ctrl+C - graceful shutdown…")
        save_checkpoint(epoch, model, optimizer,
                        record_train_loss, record_test_loss, checkpoint_path)
        raise          # ← ここで例外を再送出すると finally も確実に走る
    finally:
        # writer.close()
        save_checkpoint(epoch, model, optimizer,
                        record_train_loss, record_test_loss, checkpoint_path)
        myfunction.wirte_pkl(record_test_loss, os.path.join(result_dir, "3d_testloss"))
        myfunction.wirte_pkl(record_train_loss, os.path.join(result_dir, "3d_trainloss"))

    progress.close()     
    myfunction.send_message()
    # subprocess.run(["python", r"C:\Users\WRS\Desktop\Matsuyama\Reserch\z_test\modeltestGRUnew.py"])

    # plt.plot(range(len(record_train_loss)), record_train_loss, label="Train")
    # plt.plot(range(len(record_test_loss)), record_test_loss, label="Test")


    # plt.legend()


    # plt.xlabel("Epochs")
    # plt.ylabel("Error")
    # plt.show()
    
    # print(record_train_loss)
    # print(type(record_train_loss))   





if __name__ == '__main__':
    main()


