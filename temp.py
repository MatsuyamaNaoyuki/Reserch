import pickle
import numpy as np
from myclass import myfunction
import pandas as pd
import datetime, os, csv

def mag_data_change2(row):
    split_value = row[1].split('/')
    if len(split_value) != 9:
        split_value = split_value[1:]
    row = row[:1]
    row.extend(split_value)
    return row

magsensor = myfunction.load_pickle(r"C:\Users\shigf\Program\data\moredataset5000_0207\motor20250208_103828.pickle")
print(len(magsensor))

magsensor = magsensor[:int(len(magsensor) * 10 / 5000)]


myfunction.wirte_pkl(magsensor, "motor")

# print(len(magsensor))

# magdata = []
# for magrow in magsensor:
#     magdata.append(mag_data_change2(magrow))


# magdata.insert(0, ["time","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8","sensor9"])



# df = pd.DataFrame(magdata[1:], columns=magdata[0])

# df['sensor1'] = pd.to_numeric(df['sensor1'], errors='coerce')
# df['sensor2'] = pd.to_numeric(df['sensor2'], errors='coerce')
# df['sensor3'] = pd.to_numeric(df['sensor3'], errors='coerce')
# df['sensor4'] = pd.to_numeric(df['sensor4'], errors='coerce')
# df['sensor5'] = pd.to_numeric(df['sensor5'], errors='coerce')
# df['sensor6'] = pd.to_numeric(df['sensor6'], errors='coerce')
# df['sensor7'] = pd.to_numeric(df['sensor7'], errors='coerce')
# df['sensor8'] = pd.to_numeric(df['sensor8'], errors='coerce')
# df['sensor9'] = pd.to_numeric(df['sensor9'], errors='coerce')


# min_mag = 300
# max_mag = 800
# df = df[(df['sensor1'] > min_mag) & (df['sensor2'] > min_mag) & (df['sensor3'] > min_mag)]
# df = df[(df['sensor4'] > min_mag) & (df['sensor5'] > min_mag) & (df['sensor6'] > min_mag)]
# df = df[(df['sensor7'] > min_mag) & (df['sensor8'] > min_mag) & (df['sensor9'] > min_mag)]


# df = df[(df['sensor1'] < max_mag) & (df['sensor2'] < max_mag) & (df['sensor3'] < max_mag)]
# df = df[(df['sensor4'] < max_mag) & (df['sensor5'] < max_mag) & (df['sensor6'] < max_mag)]
# df = df[(df['sensor7'] < max_mag) & (df['sensor8'] < max_mag) & (df['sensor9'] < max_mag)]

# print(len(df))
