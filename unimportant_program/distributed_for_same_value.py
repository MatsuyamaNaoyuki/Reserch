import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.options.display.float_format = '{:.2f}'.format

csv_file_path = "1204_100data_maybeOK\\test20241205_201312.csv"
df = pd.read_csv(csv_file_path)
df = df.drop(['time', 'Mc1x', 'Mc1y', 'Mc1z'], axis = 1)
max_value = df['Mc2x'].max()



df = df[(df['Mc2x'] < 100000) & (df['Mc2y'] < 100000) & (df['Mc2z'] < 100000)]
df = df[(df['Mc3x'] < 100000) & (df['Mc3y'] < 100000) & (df['Mc3z'] < 100000)]
df = df[(df['Mc4x'] < 100000) & (df['Mc4y'] < 100000) & (df['Mc4z'] < 100000)]
df = df[(df['Mc5x'] < 100000) & (df['Mc5y'] < 100000) & (df['Mc5z'] < 100000)]

df = df.sort_values('rotate3')




df['rotate3_group'] = np.floor(df['rotate3'])
print(df)
variances = df.groupby('rotate3_group').var()
column_means = variances.mean()
# 結果を表示
print(column_means)

