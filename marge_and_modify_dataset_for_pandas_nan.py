import pickle
import numpy as np
from myclass import myfunction
import pandas as pd
import datetime, os, csv

#時刻も消して[222/222]→[222,222]
def mag_data_change(row):
    row = delete_date(row)
    if row[0] == '':
        return []
    split_value = row[0].split('/')
    # split_value = split_value[1:]
    return split_value

#時刻消さずに[時刻,"222/222"]->[時刻,222,222]
def mag_data_change2(row):
    split_value = row[1].split('/')
    if len(split_value) != 9:
        split_value = split_value[1:]
    row = row[:1]
    row.extend(split_value)
    return row

def delete_date(row):
    copyrow = row.copy()
    copyrow.pop(0)
    return copyrow

def data_marge_v1(motiondata, motordata, magsensor):
    
    margedata = []
    for motionrow in motiondata:
        one_dataset = []


        for motorrow in motordata:
            temprow = motorrow
            if motorrow[0] > motionrow[0]:
                afterdiff = motorrow[0] - motionrow[0]
                beforediff =motionrow[0] - temprow[0]
                if afterdiff < beforediff:
                    motorrow = delete_date(motorrow)
                    one_dataset.extend(motorrow)
                else:
                    temprow = delete_date(temprow)
                    one_dataset.extend(temprow)
                break
            temprow = motorrow
        for magrow in magsensor:
            temprow = magrow
            if magrow[0] > motionrow[0]:
                afterdiff = magrow[0] - motionrow[0]
                beforediff = motionrow[0] - temprow[0]
                if afterdiff < beforediff:
                    magrow = mag_data_change(magrow)
                    one_dataset.extend(magrow)
                else:
                    temprow = mag_data_change(temprow)
                    one_dataset.extend(temprow)
                break
            temprow = magrow
        
        one_dataset.insert(0,motionrow[0])
        motionrow = delete_date(motionrow)
        one_dataset.extend(motionrow)
        margedata.append(one_dataset)
    return margedata

def data_marge_v2(motiondata,motionlen, motordata, motorlen, magsensor, maglen):
    final_colums = []
    columns = ["timestamp"] + [f"motordata_{i}" for i in range(1, motorlen * 2 + 1)]
    final_colums.extend(columns)
    motor_df = pd.DataFrame(motordata, columns=columns)


    columns = ["timestamp"] + [f"magdata_{i}" for i in range(1, maglen + 1)]
    final_colums.extend(columns[1:])
    mag_df = pd.DataFrame(magsensor, columns=columns)


    columns = ["timestamp"] + [f"motiondata_{i}" for i in range(1, motionlen * 3 + 1)]
    final_colums.extend(columns[1:])
    motion_df = pd.DataFrame(motiondata, columns=columns)
    
    motion_df = motion_df.sort_values("timestamp")
    motor_df = motor_df.sort_values("timestamp")
    mag_df = mag_df.sort_values("timestamp")


    # magandmotor_df = pd.merge_asof(mag_df, motor_df, on="timestamp", direction="nearest")
    # merge_df = pd.merge_asof(magandmotor_df, motion_df, on="timestamp", direction="nearest")


    magandmotor_df = pd.merge_asof(mag_df, motor_df, on="timestamp", direction="nearest")
    merge_df = pd.merge_asof(magandmotor_df, motion_df, on="timestamp", direction="nearest")
    merge_df = merge_df[final_colums]
    print(f'全体の行数:{len(merge_df)}')

    return merge_df.values.tolist()


dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\base"
magfile = myfunction.find_pickle_files("magsensor")
magsensor = myfunction.load_pickle(magfile)
magdata = []
for magrow in magsensor:
    magdata.append(mag_data_change2(magrow))
maglen = len(magdata[0]) - 1


motorfile = myfunction.find_pickle_files("motor")
motordata = myfunction.load_pickle(motorfile)
motorlen = int((len(motordata[0]) - 1) / 2) 



motionfile = myfunction.find_pickle_files("motioncapture")
motiondata = myfunction.load_pickle(motionfile)
motionlen = int((len(motiondata[0]) - 1) /3)

  



margedata = data_marge_v2(motiondata, motionlen, motordata,motorlen, magdata, maglen)

lenname = ["time"] +\
          [f"rotate{i}" for i in range(1, motorlen+1)] + \
          [f"force{i}"  for i in range(1, motorlen+1)] + \
          [f"sensor{i}"  for i in range(1, maglen+1)] + \
          [f"Mc{i}{p}" for i in range(1, motionlen+1) for p in ("x", "y", "z")]

margedata.insert(0, lenname)

###==================================================atode

df = pd.DataFrame(margedata[1:], columns=margedata[0])


original_len = len(df)


cols_to_check = ['Mc1x', 'Mc1y', 'Mc1z',
                 'Mc2x', 'Mc2y', 'Mc2z',
                 'Mc3x', 'Mc3y', 'Mc3z',
                 'Mc4x', 'Mc4y', 'Mc4z',
                 'Mc5x', 'Mc5y', 'Mc5z']

row_mask = (df[cols_to_check] >= 100000).any(axis=1)
# 該当行数をカウント
nan_row_count = row_mask.sum()
# その行すべてをNaNに置き換え
df.loc[row_mask, :] = np.nan



print(f"motionで削除された行数: {nan_row_count}")

cols_to_check = ['force1', 'force2', 'force3']

row_mask = ((df[cols_to_check] > 10000) | (df[cols_to_check] < -2000)).any(axis=1)
# 該当行数をカウント
nan_row_count = row_mask.sum()
# その行すべてをNaNに置き換え
df.loc[row_mask, :] = np.nan



print(f"forceで削除された行数: {nan_row_count}")

min_mag = 300
max_mag = 900


df['sensor1'] = pd.to_numeric(df['sensor1'], errors='coerce')
df['sensor2'] = pd.to_numeric(df['sensor2'], errors='coerce')
df['sensor3'] = pd.to_numeric(df['sensor3'], errors='coerce')
df['sensor4'] = pd.to_numeric(df['sensor4'], errors='coerce')
df['sensor5'] = pd.to_numeric(df['sensor5'], errors='coerce')
df['sensor6'] = pd.to_numeric(df['sensor6'], errors='coerce')
df['sensor7'] = pd.to_numeric(df['sensor7'], errors='coerce')
df['sensor8'] = pd.to_numeric(df['sensor8'], errors='coerce')
df['sensor9'] = pd.to_numeric(df['sensor9'], errors='coerce')

sensor_cols = [f'sensor{i}' for i in range(1, 10)]
row_mask = (
    (df[sensor_cols] < min_mag) | (df[sensor_cols] > max_mag)
).any(axis=1)
nan_row_count = row_mask.sum()

# 該当行を NaN に置換
df.loc[row_mask, :] = np.nan
print(f"magで削除された行数: {nan_row_count}")

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
# df = df[df.index  % 10 == 0]

for col in df.columns:
    if df[col].dtype.kind in 'biufc':  # 数値列のみ対象（int, float, complex）
        col_min = df[col].min(skipna=True)
        col_max = df[col].max(skipna=True)

        row_min = df[df[col] == col_min].index[0]
        row_max = df[df[col] == col_max].index[0]

        print(f"{col}:")
        print(f"  最小値: {col_min:.6f}（行 {row_min}）")
        print(f"  最大値: {col_max:.6f}（行 {row_max}）\n")


# manual_mask = []
# df.loc[manual_mask, :] = np.nan


filename = "0818_tubefinger_hit_10kai_rerere"

now = datetime.datetime.now()
filename = filename + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
# print(df.dtypes)
# print(type(df[2][10]))

df.to_pickle(filename)
