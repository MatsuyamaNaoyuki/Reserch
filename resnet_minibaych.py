import time, os
#resnetを実装したもの
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
from myclass import myfunction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import keyboard

class ResNetRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNetRegression, self).__init__()
        self.resnet = resnet18(weights=None)

        # 最初の畳み込み層を 2D Conv に変更
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        # BatchNorm2d を BatchNorm1d に変更する必要はありません
        self.resnet.bn1 = nn.BatchNorm2d(64)

        # 出力層を回帰問題用に変更
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)  # (batch_size, input_dim) -> (batch_size, 1, input_dim, 1)
        out = self.resnet(x)
        return out


#1エポックの学習
def train(model, data_loader):
    # 今は学習時であることを明示するコード
    model.train()
    loss_mean = 0
    # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる

    for j, (x_train_data, y_train_data) in enumerate(data_loader):
        y_pred = model(x_train_data)  # 順伝播
        # print(y_pred)
        optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
        loss = criterion(y_pred, y_train_data)  # 損失を計算
        loss.backward()  # 逆伝播で勾配を計算
        optimizer.step()  # 最適化
        loss_mean += loss.item()


    loss_mean = loss_mean / (j+1)

    return loss_mean



#testの時のコード
def test(model, data_loader):
    # モデルを評価モードにする
    model.eval()
    loss_mean = 0

    for j, (x_train_data, y_train_data) in enumerate(data_loader):
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

# チェックポイントの読み込み関数
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['record_train_loss'], checkpoint['record_test_loss']



#------------------------------------------------------------------------------------------------------------------------
motor_angle = True
motor_force = False
magsensor = False

result_dir = r"sentan_morecam\angle_norandam"
data_name = r"modifydata20250122.csv"
resume_training = False  # 再開したい場合は True にする
csv = True #今後Flaseに統一
#------------------------------------------------------------------------------------------------------------------------

base_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult"
result_dir = os.path.join(base_path, result_dir)

# ハイパーパラメータ
input_dim = 4 * motor_angle + 4 * motor_force + 9 * magsensor
output_dim = 12
learning_rate = 0.001
num_epochs = 500
batch_size =256
r = 0.8

# モデルの初期化
model = ResNetRegression(input_dim=input_dim, output_dim=output_dim)

if torch.cuda.is_available():
    model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)






if len(result_dir.split(os.sep)) > 1:
    filename = os.path.dirname(result_dir)
filename = os.path.join(filename, data_name)

if csv:
    x_data,y_data = myfunction.read_csv_to_torch(filename, motor_angle, motor_force, magsensor)
else:
    x_data,y_data = myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)
x_data = x_data.to(device)
y_data = y_data.to(device)
print(len(x_data))
print(y_data[0])

x_data = (x_data - x_data.mean()) / x_data.std()
y_data = (y_data - y_data.mean()) / y_data.std()


dataset = torch.utils.data.TensorDataset(x_data,y_data)


total_size = len(dataset)
train_size = int(total_size * 0.8)  # 訓練データサイズ (80%)
test_size = total_size - train_size  # テストデータサイズ (20%)

# インデックスを固定で分割（最後の20%をテスト用に）
train_indices = list(range(0, train_size))
test_indices = list(range(train_size, total_size))

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [r,1-r])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# print(train_dataset)


start = time.time()  # 現在時刻（処理開始前）を取得



# 学習ループ


checkpoint_path = os.path.join(result_dir, "3d_checkpoint.pth")
# checkpoint_path = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\sentan_morecam\\3d_checkpoint.pth"


if resume_training and os.path.exists(checkpoint_path):


    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    record_train_loss = checkpoint['record_train_loss']
    record_test_loss = checkpoint['record_test_loss']
    print(f"Resuming training from epoch {start_epoch}.")
else:
    start_epoch = 0
    record_train_loss = []
    record_test_loss = []


try:
    for epoch in range(start_epoch, num_epochs):
        print(epoch)
        end = time.time()  # 現在時刻（処理完了後）を取得
        time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
        print(time_diff)  # 処理にかかった時間データを使用

        train_loss = train(model, train_loader)
        test_loss = test(model, test_loader)


        record_train_loss.append(train_loss)
        record_test_loss.append(test_loss)

        if epoch%10 == 0:
            # modelの保存を追加

            filename = '\\3d_model_epoch' + str(epoch)+"_"
            filename = result_dir + filename
            myfunction.save_model(model, filename)
            print(f"epoch={epoch}, train:{train_loss:.5f}, test:{test_loss:.5f}")
        if keyboard.is_pressed('q'):
            print("Training stopped by user.")
            break
except KeyboardInterrupt:
    save_checkpoint(epoch, model, optimizer, record_train_loss, record_test_loss, checkpoint_path)
    print("finish")

# myfunction.wirte_pkl(record_test_loss, "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\sentan_morecam\\3d_testloss")
# myfunction.wirte_pkl(record_train_loss, "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\sentan_morecam\\3d_trainloss")

myfunction.wirte_pkl(record_test_loss, os.path.join(result_dir, "3d_testloss"))
myfunction.wirte_pkl(record_train_loss, os.path.join(result_dir, "3d_trainloss"))
plt.plot(range(len(record_train_loss)), record_train_loss, label="Train")
plt.plot(range(len(record_test_loss)), record_test_loss, label="Test")


plt.legend()


plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()


print(record_train_loss)
print(type(record_train_loss))   

end = time.time()  # 現在時刻（処理完了後）を取得
time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
print(time_diff)  # 処理にかかった時間データを使用



