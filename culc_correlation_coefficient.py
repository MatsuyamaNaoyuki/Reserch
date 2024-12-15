import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


csv_file_path = "1204_100data_maybeOK\\test20241205_201312.csv"
df = pd.read_csv(csv_file_path)
df = df.drop(['time', 'Mc1x', 'Mc1y', 'Mc1z'], axis = 1)
df_corr = df.corr()
print(df_corr)
df_mean = df.mean()



print(df_mean)
