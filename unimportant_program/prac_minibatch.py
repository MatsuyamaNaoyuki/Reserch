import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils

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

class MakeDataset(torch.utils.data.Dataset):
    