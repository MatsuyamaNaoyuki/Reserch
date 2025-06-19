import csv, pickle
import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myclass import myfunction

def print_loss_graph(trainpath, testpath, graphname):

    traindf = pd.read_pickle(trainpath)
    traindf = pd.DataFrame(traindf)

   
    testdf = pd.read_pickle(testpath)
    testdf = pd.DataFrame(testdf)


    df_contact = pd.concat([traindf, testdf],axis=1)
    df_contact.columns = ['train_loss', 'test_loss']




    fig, ax = plt.subplots(figsize = (8.0, 6.0)) 
    df_contact.plot(ax=ax, ylim=(0, 0.01), xlim = (0,500))  # ylimを直接指定
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()

    ax.set_title(graphname,fontsize=20)
    ax.set_ylim(0,0.1)
    ax.set_xlim(0,500)
    ax.set_xlabel("epoch_number",fontsize=20)
    ax.set_ylabel("loss",fontsize=20)
    ax.set_xticklabels(xticklabels,fontsize=12)
    ax.set_yticklabels(yticklabels,fontsize=12)   
    plt.show()



trainpath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger0526\data32_stride1\3d_trainloss20250603_134521.pickle"
testpath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger0526\data32_stride1\3d_testloss20250603_134521.pickle"
graphname = "alluse"

print_loss_graph(trainpath, testpath, graphname)
