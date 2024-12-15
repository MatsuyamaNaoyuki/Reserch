import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def print_graph(x, y, labelx, labely):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(list(range(len(x))), y)
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    fig.show()
    input()




np.random.seed(2020)

_x = np.random.uniform(0, 10, 100)

x1 = np.sin(_x)
x2 = np.exp(_x / 5)
x = np.stack([x1, x2], axis=1)
y = 3 * x1 + 2 * x2 + np.random.uniform(-1,1,100)

class Net(torch.nn.Module):
    def __init__(self, inputsize, outputsize):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(inputsize,64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32,outputsize)
    
    def forward(self,x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


num_epochs = 1000

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y.reshape(-1,1)).float()



inputsize = 2
outputsize = 1
net = Net(inputsize, outputsize)
net.train()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
criterion = torch.nn.MSELoss()

epoch_loss = []
for epoch in range(num_epochs):
    outputs = net.forward(x_tensor)
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss.append(loss.data.numpy().tolist())


print_graph(epoch_loss, epoch_loss, '#epoch', 'loss')

net.eval()
_x_new = np.linspace(0, 10, 1000)
x1_new = np.sin(_x_new)
x2_new = np.exp(_x_new / 5)
x_new = np.stack([x1_new, x2_new], axis = 1)

x_new_tensor = torch.from_numpy(x_new).float()
with torch.no_grad():
    y_pred_tensor = net(x_new_tensor)

y_pred = y_pred_tensor.data.numpy()

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(_x, y)
ax.plot(_x_new, y_pred, c='orange')
ax.set_xlabel('_x')
ax.set_ylabel('y')
fig.show()
input()