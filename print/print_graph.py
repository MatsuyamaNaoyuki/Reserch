import csv, pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from myclass import myfunction


file_path = "C:\\Users\\shigf\\Program\\data\\modi_margeMc_onlymotor_tewntydata_trainloss20241218_022550.pickle"
traindf = pd.read_pickle(file_path)
traindf = pd.DataFrame(traindf)

file_path = "C:\\Users\\shigf\\Program\\data\\modi_margeMc_onlymotor_tewntydata_testloss20241218_022550.pickle"
testdf = pd.read_pickle(file_path)
testdf = pd.DataFrame(testdf)


df_contact = pd.concat([traindf, testdf],axis=1)
df_contact.columns = ['train_loss', 'test_loss']




fig, ax = plt.subplots(figsize = (8.0, 6.0)) 
df_contact.plot(ax=ax, ylim=(0, 0.05))  # ylimを直接指定
xticklabels = ax.get_xticklabels()
yticklabels = ax.get_yticklabels()

ax.set_title("motorangle",fontsize=20)
ax.set_ylim(0,0.05)
ax.set_xlabel("epoch_number",fontsize=20)
ax.set_ylabel("loss",fontsize=20)
ax.set_xticklabels(xticklabels,fontsize=12)
ax.set_yticklabels(yticklabels,fontsize=12)
plt.show()

file_path = "C:\\Users\\shigf\\Program\\data\\modi_margeMc_nomotorforce_tewntydata_trainloss20241218_172808.pickle"
traindf = pd.read_pickle(file_path)
traindf = pd.DataFrame(traindf)

file_path = "C:\\Users\\shigf\\Program\\data\\modi_margeMc_nomotorforce_tewntydata_testloss20241218_172808.pickle"
testdf = pd.read_pickle(file_path)
testdf = pd.DataFrame(testdf)


df_contact = pd.concat([traindf, testdf],axis=1)
df_contact.columns = ['train_loss', 'test_loss']


fig, ax = plt.subplots(figsize = (8.0, 6.0)) 
df_contact.plot(ax=ax, ylim=(0, 0.05))  # ylimを直接指定
xticklabels = ax.get_xticklabels()
yticklabels = ax.get_yticklabels()

ax.set_title("motorangle and magsensor",fontsize=20)
ax.set_ylim(0,0.05)
ax.set_xlabel("epoch_number",fontsize=20)
ax.set_ylabel("loss",fontsize=20)
ax.set_xticklabels(xticklabels,fontsize=12)
ax.set_yticklabels(yticklabels,fontsize=12)
plt.show()

file_path = "C:\\Users\\shigf\\Program\\data\\modi_margeMc_alldata_tewntydata_trainloss20241218_024252.pickle"
traindf = pd.read_pickle(file_path)
traindf = pd.DataFrame(traindf)

file_path = "C:\\Users\\shigf\\Program\\data\\modi_margeMc_alldata_tewntydata_testloss20241218_024252.pickle"
testdf = pd.read_pickle(file_path)
testdf = pd.DataFrame(testdf)

df_contact = pd.concat([traindf, testdf],axis=1)
df_contact.columns = ['train_loss', 'test_loss']



# print(df_contact)

fig, ax = plt.subplots(figsize=(8.0, 6.0)) 
df_contact.plot(ax=ax, ylim=(0, 0.05))  # ylimを直接指定

xticklabels = ax.get_xticklabels()
yticklabels = ax.get_yticklabels()

ax.set_title("alldata", fontsize=20)
ax.set_xlabel("epoch_number", fontsize=20)
ax.set_ylabel("loss", fontsize=20)
ax.set_xticklabels(xticklabels, fontsize=12)
ax.set_yticklabels(yticklabels, fontsize=12)

plt.show()