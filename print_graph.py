import csv, pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from myclass import myfunction


file_path = "C:\\Users\\shigf\\Program\\data\\modi_margeMc_alldata_tewntydata_trainloss20241218_024252.pickle"
traindf = pd.read_pickle(file_path)
traindf = pd.DataFrame(traindf)

file_path = "C:\\Users\\shigf\\Program\\data\\modi_margeMc_alldata_tewntydata_testloss20241218_024252.pickle"
testdf = pd.read_pickle(file_path)
testdf = pd.DataFrame(testdf)


# file_path = "1204_100data_maybeOK\\modifying_all\\dataset_1209.csv"
# dataset = pd.read_pickle(file_path)
# dataset = pd.DataFrame(testdf)

df_contact = pd.concat([traindf, testdf],axis=1)
df_contact.columns = ['train_loss', 'test_loss']



# print(df_contact)

fig, ax = plt.subplots() 
df_contact.plot(ax = ax)


ax.set_title("all_data")
ax.set_ylim (0,0.05)
ax.set_xlabel("epoch_number")
ax.set_ylabel("loss")

plt.show()