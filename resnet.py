import time
#resnetを実装したもの
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from myclass import myfunction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# ハイパーパラメータ
input_dim = 17
output_dim = 5
learning_rate = 0.001
num_epochs = 100

# モデルの初期化
model = ResNetRegression(input_dim=input_dim, output_dim=output_dim)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

filename = "dami-.csv"
x_data,y_data = myfunction.read_csv_to_torch(filename)
x_data = x_data.to(device)
y_data = y_data.to(device)
x_data = (x_data - x_data.mean()) / x_data.std()
y_data = (y_data - y_data.mean()) / y_data.std()



train_size = int(0.8 * len(x_data))
test_size = len(x_data) - train_size

# 訓練用とテスト用に分割
x_train_data = x_data[:train_size]
y_train_data = y_data[:train_size]
x_test_data = x_data[train_size:]
y_test_data = y_data[train_size:]



start = time.time()  # 現在時刻（処理開始前）を取得



# 学習ループ
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train_data)
    loss = criterion(y_pred, y_train_data)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', end='  ')
        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test_data)
            test_loss = criterion(y_test_pred, y_test_data)
            print(f'Test Loss: {test_loss.item():.4f}')


# テスト
model.eval()
with torch.no_grad():
    y_test_pred = model(x_test_data)
    test_loss = criterion(y_test_pred, y_test_data)
    print(f'Test Loss: {test_loss.item():.4f}')

    
end = time.time()  # 現在時刻（処理完了後）を取得
time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
print(time_diff)  # 処理にかかった時間データを使用