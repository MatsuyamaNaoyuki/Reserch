import csv, pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from myclass import myfunction
# import japanize_matplotlib



file_path = "C:\\Users\\shigf\\Program\\DXhub\\motioncapture20241205_005155.pickle"
dataset = pd.read_pickle(file_path)
dataset = pd.DataFrame(dataset)

print(dataset.columns)
dataset = dataset.loc[:, 10]

# print(dataset.mean())
# print(dataset.std())


fig, ax = plt.subplots() 
dataset.hist(ax = ax, bins = 100)
# xticklabels = ax.get_xticklabels()
# yticklabels = ax.get_yticklabels()
# ax.set_xticklabels(xticklabels,fontsize=10)
# ax.set_yticklabels(yticklabels,fontsize=10)
# ax.set_xlabel("value", fontsize = 10)
# ax.set_ylabel("frequency", fontsize = 10)
# ax.set_title("before", fontsize = 10)
# fig.set_size_inches(18.5, 10.5)
# fig.tight_layout()
plt.show()

# dataset = (dataset - dataset.mean()) / dataset.std()


# fig, ax = plt.subplots() 
# dataset.hist(ax = ax, bins = 100)
# xticklabels = ax.get_xticklabels()
# yticklabels = ax.get_yticklabels()
# ax.set_xticklabels(xticklabels,fontsize=50)
# ax.set_yticklabels(yticklabels,fontsize=50)
# ax.set_xlabel("value", fontsize = 50)
# ax.set_ylabel("frequency", fontsize = 50)
# ax.set_title("標準化後", fontsize = 50)
# fig.set_size_inches(18.5, 10.5)
# fig.tight_layout()
# plt.show()