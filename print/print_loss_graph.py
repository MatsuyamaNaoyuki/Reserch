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
    df_contact.plot(ax=ax, ylim=(0, 0.01), xlim = (0,300))  # ylimを直接指定
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()

    ax.set_title(graphname,fontsize=20)
    ax.set_ylim(0,0.02)
    ax.set_xlim(0,500)
    ax.set_xlabel("epoch_number",fontsize=20)
    ax.set_ylabel("loss",fontsize=20)
    ax.set_xticklabels(xticklabels,fontsize=12)
    ax.set_yticklabels(yticklabels,fontsize=12)   
    plt.show()


filepath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\nomotor4"
trainpath = myfunction.find_pickle_files("trainloss", filepath)
testpath = myfunction.find_pickle_files("testloss", filepath)
graphname = "alluse"

print_loss_graph(trainpath, testpath, graphname)
