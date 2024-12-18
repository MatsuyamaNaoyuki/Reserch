import time
#resnetを実装したもの
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from myclass import myfunction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

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




# ハイパーパラメータ
input_dim = 13
output_dim = 12
learning_rate = 0.001
num_epochs = 500
batch_size =128
r = 0.8

# モデルの初期化
model = ResNetRegression(input_dim=input_dim, output_dim=output_dim)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

filename = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\dataset_margemag_tewnty.csv"
x_data,y_data = myfunction.read_csv_to_torch(filename)
x_data = x_data.to(device)
y_data = y_data.to(device)
print(x_data[0])
print(y_data[0])

x_data = (x_data - x_data.mean()) / x_data.std()
y_data = (y_data - y_data.mean()) / y_data.std()


dataset = torch.utils.data.TensorDataset(x_data,y_data)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [r,1-r])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# print(train_dataset)


start = time.time()  # 現在時刻（処理開始前）を取得



# 学習ループ
record_train_loss = []
record_test_loss = []

for epoch in range(num_epochs):
    print(epoch)
    train_loss = train(model, train_loader)
    test_loss = test(model, test_loader)


    record_train_loss.append(train_loss)
    record_test_loss.append(test_loss)

    if epoch%10 == 0:
        # modelの保存を追加
        dir_name = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\nomotorforce_tewntydata\\"
        filename = 'modi_margeMc_nomotorforce_tewntydata_model_epoch' + str(epoch)+"_"
        filename = dir_name + filename
        myfunction.save_model(model, filename)
        print(f"epoch={epoch}, train:{train_loss:.5f}, test:{test_loss:.5f}")



myfunction.wirte_pkl(record_test_loss, "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\nomotorforce_tewntydata\\modi_margeMc_nomotorforce_tewntydata_testloss")
myfunction.wirte_pkl(record_train_loss, "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\nomotorforce_tewntydata\\modi_margeMc_nomotorforce_tewntydata_trainloss")

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



