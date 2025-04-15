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

class ResNetMultiTask(nn.Module):
    def __init__(self, input_dim, output_dim_regression, output_dim_classification):
        super(ResNetMultiTask, self).__init__()
        self.resnet = resnet18(weights=None)

        # 入力の畳み込み層を変更
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.resnet.bn1 = nn.BatchNorm2d(64)

        # 出力層を2つに分ける（回帰用 + 分類用）
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # ResNetの全結合層を削除
        self.fc_regression = nn.Linear(num_features, output_dim_regression)  # 12個の回帰出力
        self.fc_classification = nn.Linear(num_features, output_dim_classification)  # 1個の分類出力（0 or 1）

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)  # (batch_size, input_dim) -> (batch_size, 1, input_dim, 1)
        features = self.resnet(x)
        regression_output = self.fc_regression(features)
        classification_output = self.fc_classification(features)  # 分類は確率値
        return regression_output, classification_output  # 2つの出力を返す


#1エポックの学習
def train(model, data_loader):
    model.train()
    loss_mean = 0
    loss_reg_mean = 0
    loss_cls_mean = 0

    for j, (x_train_data, y_train_regression, y_train_classification) in enumerate(data_loader):
        y_pred_regression, y_pred_classification = model(x_train_data)  # 順伝播

        optimizer.zero_grad()  # 勾配を初期化
        loss_reg = criterion_regression(y_pred_regression, y_train_regression)  # 回帰損失
        loss_cls = criterion_classification(y_pred_classification, y_train_classification)  # 分類損失
        loss = loss_reg + loss_cls  # 合計の損失

        loss.backward()  # 逆伝播で勾配を計算
        optimizer.step()  # 最適化

        loss_mean += loss.item()
        loss_reg_mean += loss_reg.item()
        loss_cls_mean += loss_cls.item()

    loss_mean /= (j+1)
    loss_reg_mean /= (j+1)
    loss_cls_mean /= (j+1)

    return loss_mean, loss_reg_mean, loss_cls_mean



#testの時のコード
def test(model, data_loader):
    model.eval()
    loss_mean = 0
    loss_reg_mean = 0
    loss_cls_mean = 0

    with torch.no_grad():
        for j, (x_test_data, y_test_regression, y_test_classification) in enumerate(data_loader):
            y_pred_regression, y_pred_classification = model(x_test_data)

            loss_reg = criterion_regression(y_pred_regression, y_test_regression)
            loss_cls = criterion_classification(y_pred_classification, y_test_classification)
            loss = loss_reg + loss_cls  # 合計の損失

            loss_mean += loss.item()
            loss_reg_mean += loss_reg.item()
            loss_cls_mean += loss_cls.item()

    loss_mean /= (j+1)
    loss_reg_mean /= (j+1)
    loss_cls_mean /= (j+1)

    return loss_mean, loss_reg_mean, loss_cls_mean


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

def save_test(test_dataset, result_dir):


    test_indices = test_dataset.indices  # 添え字のみ取得
# **保存パス**
    test_indices_path = os.path.join(result_dir, "test_indices")

    # **pickle ファイルとして保存**
    myfunction.wirte_pkl(test_indices, test_indices_path)

#---------------------------------------------------------------------------------- --------------------------------------
motor_angle = True
motor_force = True
magsensor = True
original_result_dir = r"mixhit"
data_name = r"hit_mix20250224_150035.pickle"
resume_training = True  # 再開したい場合は True にする
csv = False#今後Flaseに統一
hitting = True
#------------------------------------------------------------------------------------------------------------------------

base_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult"
result_dir = os.path.join(base_path, original_result_dir)

# ハイパーパラメータ
input_dim = 4 * motor_angle + 4 * motor_force + 9 * magsensor
output_dim_regression = 12  # 回帰出力
output_dim_classification = 1  # 分類出力（0 or 1）
learning_rate = 0.001
num_epochs = 500
batch_size =256
r = 0.8

# モデルの初期化
model = ResNetMultiTask(input_dim, output_dim_regression, output_dim_classification)

if torch.cuda.is_available():
    model.cuda()
criterion_regression = nn.MSELoss()
criterion_classification = nn.BCEWithLogitsLoss()  # 分類用（バイナリクロスエントロ
optimizer = optim.Adam(model.parameters(), lr=learning_rate)






if len(original_result_dir.split(os.sep)) > 1:
    filename = os.path.dirname(result_dir)
    filename = os.path.join(filename, data_name)
else:
    filename = os.path.join(result_dir, data_name)
if csv:
    x_data,y_data = myfunction.read_csv_to_torch(filename, motor_angle, motor_force, magsensor)
else:
    x_data,y_data = myfunction.read_pickle_to_torch(filename, motor_angle, motor_force, magsensor)


x_data = x_data.to(device)
y_data = y_data.to(device)
print(len(x_data))
print(y_data[0])

y_classification = y_data[:, -1].unsqueeze(1)  # 形状を (batch_size, 1) に整える
y_regression = y_data[:, :-1] 

x_mean = x_data.mean(dim=0, keepdim=True)
x_std = x_data.std(dim=0, keepdim=True)
y_mean = y_regression.mean(dim=0, keepdim=True)
y_std = y_regression.std(dim=0, keepdim=True)

scaler_data = {
    'x_mean': x_mean.cpu().numpy(),  # GPUからCPUへ移動してnumpy配列へ変換
    'x_std': x_std.cpu().numpy(),
    'y_mean': y_mean.cpu().numpy(),
    'y_std': y_std.cpu().numpy()
}

  # 0 or 1 に変換


x_data = (x_data - x_mean) / x_std
y_regression = (y_regression - y_mean) / y_std
x_data = x_data.nan_to_num(0.0)
scaler_pass = os.path.join(result_dir, "scaler")
# pickleファイルとして保存
myfunction.wirte_pkl(scaler_data, scaler_pass)

dataset = torch.utils.data.TensorDataset(x_data, y_regression, y_classification)

# 訓練データとテストデータに分割
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [r, 1-r])

# DataLoader 作成
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


save_test(test_dataset,result_dir)




start = time.time()  # 現在時刻（処理開始前）を取得



# 学習ループ


checkpoint_path = os.path.join(result_dir, "3d_checkpoint.pth")



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

        train_loss, train_reg_loss, train_cls_loss = train(model, train_loader)
        test_loss, test_reg_loss, test_cls_loss = test(model, test_loader)

        record_train_loss.append(train_loss)
        record_test_loss.append(test_loss)

        if epoch%10 == 0:
            # modelの保存を追加

            filename = '\\3d_model_epoch' + str(epoch)+"_"
            filename = result_dir + filename
            myfunction.save_model(model, filename)
            print(f"Epoch {epoch}: Train Loss: {train_loss:.5f} (Reg: {train_reg_loss:.5f}, Cls: {train_cls_loss:.5f})")
            print(f"          Test Loss : {test_loss:.5f} (Reg: {test_reg_loss:.5f}, Cls: {test_cls_loss:.5f})")
        if keyboard.is_pressed('q'):
            print("Training stopped by user.")
            break
except KeyboardInterrupt:
    save_checkpoint(epoch, model, optimizer, record_train_loss, record_test_loss, checkpoint_path)
    print("finish")



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



