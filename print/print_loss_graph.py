import csv, pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from myclass import myfunction


def plot_loss(trainloss, testloss):
    # x軸のデータを作成（リストの長さに基づく）
    epochs = list(range(1, len(trainloss) + 1))

    # グラフをプロット
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, trainloss, label="Train Loss")
    plt.plot(epochs, testloss, label="Test Loss")

    # グラフの装飾
    plt.ylim(0,0.001)
    plt.xlim(0,500)  
    plt.title("angle_and_mag")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

file_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\sentan_morecam\all_use\3d_trainloss20250123_162722.pickle"
trainloss = myfunction.load_pickle(file_path)

file_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\sentan_morecam\all_use\3d_testloss20250123_162722.pickle"
testloss = myfunction.load_pickle(file_path)

print(type(testloss))


plot_loss(trainloss, testloss)

# df_contact = pd.concat([traindf, testdf],axis=1)
# df_contact.columns = ['train_loss', 'test_loss']




# fig, ax = plt.subplots(figsize = (8.0, 6.0)) 
# df_contact.plot(ax=ax, ylim=(0, 1.0))  # ylimを直接指定
# xticklabels = ax.get_xticklabels()
# yticklabels = ax.get_yticklabels()

# ax.set_title("motorangle",fontsize=20)
# ax.set_ylim(0,0.05)
# ax.set_xlabel("epoch_number",fontsize=20)
# ax.set_ylabel("loss",fontsize=20)
# ax.set_xticklabels(xticklabels,fontsize=12)
# ax.set_yticklabels(yticklabels,fontsize=12)
# plt.show()

