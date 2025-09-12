import pickle
import numpy as np
from myclass import myfunction
import pandas as pd
import datetime, os, csv



#時刻消さずに[時刻,"222/222"]->[時刻,222,222]
def mag_data_change2(row):
    split_value = row[1].split('/')
    if len(split_value) != 9:
        split_value = split_value[1:]
    row = row[:1]
    row.extend(split_value)
    return row


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

def modify_Mc(df):
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
    
    return df

def modify_force(df):
    cols_to_check = ['force1', 'force2', 'force3']
    row_mask = ((df[cols_to_check] > 10000) | (df[cols_to_check] < -2000)).any(axis=1)
    # 該当行数をカウント
    nan_row_count = row_mask.sum()
    # その行すべてをNaNに置き換え
    df.loc[row_mask, :] = np.nan
    print(f"forceで削除された行数: {nan_row_count}")
    return df
    
def modify_mag(df):
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
    
    return df

def load_df(magname, motorname, motionname):
    magfile = myfunction.find_pickle_files(magname, base_path)
    magsensor = myfunction.load_pickle(magfile)
    magdata = []
    for magrow in magsensor:
        magdata.append(mag_data_change2(magrow))
    maglen = len(magdata[0]) - 1


    motorfile = myfunction.find_pickle_files(motorname, base_path)
    motordata = myfunction.load_pickle(motorfile)
    motorlen = int((len(motordata[0]) - 1) / 2) 



    motionfile = myfunction.find_pickle_files(motionname, base_path)
    motiondata = myfunction.load_pickle(motionfile)
    motionlen = int((len(motiondata[0]) - 1) /3)

    
    margedata = data_marge_v2(motiondata, motionlen, motordata,motorlen, magdata, maglen)

    lenname = ["time"] +\
            [f"rotate{i}" for i in range(1, motorlen+1)] + \
            [f"force{i}"  for i in range(1, motorlen+1)] + \
            [f"sensor{i}"  for i in range(1, maglen+1)] + \
            [f"Mc{i}{p}" for i in range(1, motionlen+1) for p in ("x", "y", "z")]



    df = pd.DataFrame(margedata, columns=lenname)
    return df

# ----------------------------------------------------------------------------------------------
base_path = r"C:\Users\shigf\Program\data\0903_tubefinger_re3\nohit_1500kai"
filename = "0910_tubefinger_nohit_1500kai_re3"
kijun = True
kijunname = "kijuntrain"
# -----------------------------------------------------------------------------------------------

kijundf = load_df("mag_1_", "motor_1_", "mc_1_")
kijundf = modify_Mc(kijundf)
kijundf = modify_force(kijundf)
kijundf = modify_mag(kijundf)
kijunlen = len(kijundf)



df = load_df("magsensor202", "motor202", "motioncapture202")
df = modify_Mc(df)
df = modify_force(df)
df = modify_mag(df)

df = df.iloc[kijunlen:]
df = df.reset_index()




for col in df.columns:
    if df[col].dtype.kind in 'biufc':  # 数値列のみ対象（int, float, complex）
        col_min = df[col].min(skipna=True)
        col_max = df[col].max(skipna=True)

        row_min = df[df[col] == col_min].index[0]
        row_max = df[df[col] == col_max].index[0]

        print(f"{col}:")
        print(f"  最小値: {col_min:.6f}（行 {row_min}）")
        print(f"  最大値: {col_max:.6f}（行 {row_max}）\n")




myfunction.print_val(df)

parent_dir = os.path.dirname(base_path)
filename = os.path.join(parent_dir, filename)
myfunction.wirte_pkl(df, filename)

if kijun:
    filename = os.path.join(parent_dir, "kijuntest")
    myfunction.wirte_pkl(kijundf, filename)
    