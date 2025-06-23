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

#1エポックの学習
def train(model, data_loader):
    # 今は学習時であることを明示するコード
    model.train()
    loss_mean = 0
    # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる

    for j, (x_train_data, y_train_data) in enumerate(data_loader):
        x_train_data = x_train_data.to(device, non_blocking=True)
        y_train_data = y_train_data.to(device, non_blocking=True)
        y_pred = model(x_train_data)  # 順伝播
        # print(y_pred)
        optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
        loss = criterion(y_pred, y_train_data)  # 損失を計算
        loss.backward()  # 逆伝播で勾配を計算
        optimizer.step()  # 最適化
        loss_mean += loss.item()


    loss_mean = loss_mean / (j+1)

    return loss_mean

def make_sequence_tensor_stride(x, y, typedf,L, stride):
    typedf = typedf.tolist()
    typedf.insert(0, 0)
    total_span = (L - 1) * stride

    seq_x, seq_y = [], []

    for i in range(len(typedf) - 1):
        start = typedf[i] + total_span
        end = typedf[i + 1]
        if end <= start:
            continue

        # j の全体リスト
        js = torch.arange(start, end, device=x.device)

        # indices テンソルをまとめて作成：(len(js), L)
        relative_indices = torch.arange(L-1, -1, -1, device=x.device) * stride
        indices = js.unsqueeze(1) - relative_indices  # shape: (num_seq, L)

        # <<< ここで indices を表示 >>>
        # print(f"[Group {i}] indices shape: {indices.shape}")
        # print(indices)

        # x と y を一括取得
        x_seq = x[indices]  # shape: (num_seq, L, D)
        y_seq = y[js]       # shape: (num_seq, D_out)

        seq_x.append(x_seq)
        seq_y.append(y_seq)

    seq_x = torch.cat(seq_x, dim=0)
    seq_y = torch.cat(seq_y, dim=0)

    return torch.utils.data.TensorDataset(seq_x, seq_y)

def test(model, data_loader):
    # モデルを評価モードにする
    model.eval()
    loss_mean = 0

    for j, (x_train_data, y_train_data) in enumerate(data_loader):
        x_train_data = x_train_data.to(device, non_blocking=True)
        y_train_data = y_train_data.to(device, non_blocking=True)
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

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['record_train_loss'], checkpoint['record_test_loss']

def save_test(test_dataset, result_dir):


    test_indices = test_dataset.indices  # 添え字のみ取得
# **保存パス**
    test_indices_path = os.path.join(result_dir, "test_indices")

    # **pickle ファイルとして保存**
    myfunction.wirte_pkl(test_indices, test_indices_path)



#---------------------------------------------------------------------------------- --------------------------------------
motor_angle = False
motor_force = True
magsensor = True
L = 32 
stride = 1

result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\nohit_noangle_data32stride1"
filename = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\no_hit_for_train.pickle"
resume_training = False   # 再開したい場合は True にする


#------------------------------------------------------------------------------------------------------------------------



# ハイパーパラメータ
input_dim = 4 * motor_angle + 4 * motor_force + 9 * magsensor
output_dim = 12
learning_rate = 0.001
num_epochs = 500
batch_size =128
r = 0.8


# モデルの初期化
model = ResNetGRU(input_dim=input_dim,output_dim=output_dim, hidden=128)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


x_data,y_data, typedf = myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
print(len(x_data))
print(y_data[0])


x_mean = x_data.mean(dim=0, keepdim=True)
x_std = x_data.std(dim=0, keepdim=True)
y_mean = y_data.mean(dim=0, keepdim=True)
y_std = y_data.std(dim=0, keepdim=True)

scaler_data = {
    'x_mean': x_mean.cpu().numpy(),  # GPUからCPUへ移動してnumpy配列へ変換
    'x_std': x_std.cpu().numpy(),
    'y_mean': y_mean.cpu().numpy(),
    'y_std': y_std.cpu().numpy()
}
x_data = (x_data - x_data.mean(dim=0, keepdim=True)) / x_data.std(dim=0, keepdim=True)
y_data = (y_data - y_data.mean(dim=0, keepdim=True)) / y_data.std(dim=0, keepdim=True)
scaler_pass = os.path.join(result_dir, "scaler")

type_end_list = myfunction.get_type_change_end(typedf)
seq_dataset = make_sequence_tensor_stride(x_data, y_data,type_end_list, L, stride)

checkpoint_path = os.path.join(result_dir, "3d_checkpoint.pth")



if resume_training and os.path.exists(checkpoint_path):
    test_indices_path = myfunction.find_pickle_files("test_indices", result_dir)
    test_indices = myfunction.load_pickle(test_indices_path)
    all_indices = set(range(len(seq_dataset)))
    test_indices = set(test_indices)  # Set に変換（高速化）
    train_indices = list(all_indices - test_indices)  # 差分を取る
    train_dataset = Subset(seq_dataset, train_indices)
    test_dataset = Subset(seq_dataset, list(test_indices))  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    record_train_loss = checkpoint['record_train_loss']
    record_test_loss = checkpoint['record_test_loss']
    print(f"Resuming training from epoch {start_epoch}.")
else:
    train_dataset, test_dataset = torch.utils.data.random_split(seq_dataset, [r,1-r],generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    myfunction.wirte_pkl(scaler_data, scaler_pass)
    save_test(test_dataset,result_dir)
    start_epoch = 0
    record_train_loss = []
    record_test_loss = []







progress = tqdm(total=num_epochs,initial=start_epoch,desc ="Epoch")
try:
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, train_loader)
        test_loss  = test(model , test_loader)
        record_train_loss.append(train_loss)
        record_test_loss.append(test_loss)

        # 10 エポックごとにログを出す
        if epoch % 10 == 0:
            filename = os.path.join(result_dir, f"3d_model_epoch{epoch}_")
            myfunction.save_model(model, filename)
            tqdm.write(f"[{epoch}] train={train_loss:.5f} test={test_loss:.5f}")

        progress.update(1)           # ★ これでバーが 1 目盛り進む
except KeyboardInterrupt:
    print("\n[INFO] Detected Ctrl+C - graceful shutdown…")
    save_checkpoint(epoch, model, optimizer,
                    record_train_loss, record_test_loss, checkpoint_path)
    raise          # ← ここで例外を再送出すると finally も確実に走る
finally:
    save_checkpoint(epoch, model, optimizer,
                    record_train_loss, record_test_loss, checkpoint_path)
    myfunction.wirte_pkl(record_test_loss, os.path.join(result_dir, "3d_testloss"))
    myfunction.wirte_pkl(record_train_loss, os.path.join(result_dir, "3d_trainloss"))

progress.close()     

plt.plot(range(len(record_train_loss)), record_train_loss, label="Train")
plt.plot(range(len(record_test_loss)), record_test_loss, label="Test")


plt.legend()


plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()


# print(record_train_loss)
# print(type(record_train_loss))   





