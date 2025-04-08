import pickle
import numpy as np
from myclass import myfunction
import pandas as pd
import matplotlib.pyplot as plt
import datetime, os, csv


#時刻消さずに[時刻,"222/222"]->[時刻,222,222]
def mag_data_change2(row):
    split_value = row[1].split('/')
    if len(split_value) != 9:
        split_value = split_value[1:]
    row = row[:1]
    row.extend(split_value)
    return row

def data_marge_v2(motiondata, motordata, magsensor):
    final_colums = []

    num_cols = len(motordata[0])
    columns = ["timestamp"] + [f"motordata_{i}" for i in range(1, num_cols)] 
    final_colums.extend(columns)
    motor_df = pd.DataFrame(motordata, columns=columns)


    columns = ["timestamp"] + [f"magdata_{i}" for i in range(1, 10)]
    final_colums.extend(columns[1:])
    mag_df = pd.DataFrame(magsensor, columns=columns)


    columns = ["timestamp"] + [f"motiondata_{i}" for i in range(1, 16)]
    final_colums.extend(columns[1:])
    motion_df = pd.DataFrame(motiondata, columns=columns)
    
    motion_df = motion_df.sort_values("timestamp")
    motor_df = motor_df.sort_values("timestamp")
    mag_df = mag_df.sort_values("timestamp")


    motionandmotor_df = pd.merge_asof(motion_df, motor_df, on="timestamp", direction="nearest")
    merge_df = pd.merge_asof(motionandmotor_df, mag_df, on="timestamp", direction="nearest")
    merge_df = merge_df[final_colums]

    return merge_df.values.tolist()

def replace_short_neg_sequences(series, max_value, min_value):
    limit_length=8
    values = series.values
    start = None  # 連続部分の開始位置を記録

    for i in range(len(values)):
        if values[i] == min_value:

            if start is None:
                # print(f"i = {i}")
                start = i  # 連続の開始位置
        else:
            if start is not None:
                # print(f"start = {start}")
                # print(f"i = {i}")
                length = i - start
                # print(f"length = {length}")
                if length <= limit_length:
                    # print(values[start - 3:i + 3])
                    values[start:i] = max_value  # 置き換え
                    # print(values[start - 3:i + 3])
                start = None  # 次の連続部分を探す
                # print("---------------------------------------")
            

    # 最後の部分の処理（連続が終わらない場合）
    if start is not None:
        length = len(values) - start
        if length <= limit_length:
            values[start:] = max_value

    return pd.Series(values, index=series.index)



def change_force_to_2_value(series):
    min_value = 0
    max_value = 1
    window_size = 10
    smoothed_col = series.rolling(window=window_size, min_periods=1).mean()
    diff = smoothed_col.sub(smoothed_col.shift(2)).abs()
    two_val = diff.apply(lambda x: max_value if x >= 4 else min_value)
    two_val = replace_short_neg_sequences(two_val, max_value, min_value)
    two_val = replace_short_neg_sequences(two_val, min_value, max_value)
    return two_val

def modify_mag(df):
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
    return df

def modify_motion(df):
    df = df[(df['Mc2x'] < 100000) & (df['Mc2y'] < 100000) & (df['Mc2z'] < 100000)]
    df = df[(df['Mc3x'] < 100000) & (df['Mc3y'] < 100000) & (df['Mc3z'] < 100000)]
    df = df[(df['Mc4x'] < 100000) & (df['Mc4y'] < 100000) & (df['Mc4z'] < 100000)]
    df = df[(df['Mc5x'] < 100000) & (df['Mc5y'] < 100000) & (df['Mc5z'] < 100000)]
    
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

    
    

magfile = myfunction.find_pickle_files("magsensor")
magsensor = myfunction.load_pickle(magfile)

motorfile = myfunction.find_pickle_files("motor")
motordata = myfunction.load_pickle(motorfile)

motionfile = myfunction.find_pickle_files("motioncapture")
motiondata = myfunction.load_pickle(motionfile)


  
magdata = []
for magrow in magsensor:
    magdata.append(mag_data_change2(magrow))

if len(motordata[0]) == 10:
    hittingisTrue = True
else:
    hittingisTrue = False
    
if hittingisTrue:
    motor = pd.DataFrame(motordata)
    two_val = change_force_to_2_value(motor.iloc[:, 9])
    motor.loc[:, "Flag"] = two_val
    motor = motor.drop(9, axis=1)
    motordata = motor.values.tolist()
    




    


margedata = data_marge_v2(motiondata, motordata, magdata)


if hittingisTrue:
    margedata.insert(0, ["time","rotate1","rotate2","rotate3","rotate4","force1","force2","force3","force4","Hitornot","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8","sensor9","Mc1x","Mc1y","Mc1z","Mc2x","Mc2y","Mc2z","Mc3x","Mc3y","Mc3z","Mc4x","Mc4y","Mc4z","Mc5x","Mc5y","Mc5z"])
    df = pd.DataFrame(margedata[1:], columns=margedata[0])
    col_to_move = "Hitornot" # 後ろに移動したい列名
    df[col_to_move] = df.pop(col_to_move)  # pop() で取り出して最後に追加

else:
    margedata.insert(0, ["time","rotate1","rotate2","rotate3","rotate4","force1","force2","force3","force4","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8","sensor9","Mc1x","Mc1y","Mc1z","Mc2x","Mc2y","Mc2z","Mc3x","Mc3y","Mc3z","Mc4x","Mc4y","Mc4z","Mc5x","Mc5y","Mc5z"])
    df = pd.DataFrame(margedata[1:], columns=margedata[0])

print(df)




df = modify_mag(df)
df = modify_motion(df)
print(df.columns)




# # df = df.head(8670)
df = df[df.index  % 10 == 0]

filename = "with_hit"

now = datetime.datetime.now()
filename = filename + now.strftime('%Y%m%d_%H%M%S') + '.pickle'


df.to_pickle(filename)
