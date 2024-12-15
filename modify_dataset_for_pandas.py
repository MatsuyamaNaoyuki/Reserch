import csv, pickle
from myclass import myfunction
import pandas as pd
import matplotlib.pyplot as plt


# CSVファイルを開く



data = []
file_path = "1204_100data_maybeOK\\margedata_formotioncapture20241215_224821.pickle"


data = pd.read_pickle(file_path)

df = pd.DataFrame(data[1:], columns=data[0])


df = df[(df['Mc2x'] < 100000) & (df['Mc2y'] < 100000) & (df['Mc2z'] < 100000)]
df = df[(df['Mc3x'] < 100000) & (df['Mc3y'] < 100000) & (df['Mc3z'] < 100000)]
df = df[(df['Mc4x'] < 100000) & (df['Mc4y'] < 100000) & (df['Mc4z'] < 100000)]
df = df[(df['Mc5x'] < 100000) & (df['Mc5y'] < 100000) & (df['Mc5z'] < 100000)]

min_mag = 300
max_mag = 800


df['sensor1'] = pd.to_numeric(df['sensor1'], errors='coerce')
df['sensor2'] = pd.to_numeric(df['sensor2'], errors='coerce')
df['sensor3'] = pd.to_numeric(df['sensor3'], errors='coerce')
df['sensor4'] = pd.to_numeric(df['sensor4'], errors='coerce')
df['sensor5'] = pd.to_numeric(df['sensor5'], errors='coerce')
df['sensor6'] = pd.to_numeric(df['sensor6'], errors='coerce')
df['sensor7'] = pd.to_numeric(df['sensor7'], errors='coerce')
df['sensor8'] = pd.to_numeric(df['sensor8'], errors='coerce')
df['sensor9'] = pd.to_numeric(df['sensor9'], errors='coerce')

df = df[(df['sensor1'] > min_mag) & (df['sensor2'] > min_mag) & (df['sensor3'] > min_mag)]
df = df[(df['sensor4'] > min_mag) & (df['sensor5'] > min_mag) & (df['sensor6'] > min_mag)]
df = df[(df['sensor7'] > min_mag) & (df['sensor8'] > min_mag) & (df['sensor9'] > min_mag)]


df = df[(df['sensor1'] < max_mag) & (df['sensor2'] < max_mag) & (df['sensor3'] < max_mag)]
df = df[(df['sensor4'] < max_mag) & (df['sensor5'] < max_mag) & (df['sensor6'] < max_mag)]
df = df[(df['sensor7'] < max_mag) & (df['sensor8'] < max_mag) & (df['sensor9'] < max_mag)]

df['Mc2x'] = df['Mc2x'] - df['Mc1x']
df['Mc2y'] = df['Mc2y'] - df['Mc1y']
df['Mc2z'] = df['Mc2z'] - df['Mc1z']
df['Mc3x'] = df['Mc3x'] - df['Mc1x']
df['Mc3y'] = df['Mc3y'] - df['Mc1y']
df['Mc3z'] = df['Mc3z'] - df['Mc1z']
df['Mc4x'] = df['Mc4x'] - df['Mc1x']
df['Mc4y'] = df['Mc4y'] - df['Mc1y']
df['Mc4z'] = df['Mc4z'] - df['Mc1z']
df['Mc5x'] = df['Mc5x'] - df['Mc1x']
df['Mc5y'] = df['Mc5y'] - df['Mc1y']
df['Mc5z'] = df['Mc5z'] - df['Mc1z']

df = df.drop(columns=['Mc1x'])
df = df.drop(columns=['Mc1y'])
df = df.drop(columns=['Mc1z'])

# df = df.head(8670)
# df = df[df.index % 10 == 0]

df.to_csv('dataset_1213_margeMC_handreddata.csv', index=False)
# print(df.columns)


# myfunction.wirte_csv(, "1204_100data_maybeOK\\test")