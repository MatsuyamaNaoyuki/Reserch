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

def data_marge_v2(motiondata, motordata, magsensor):
    final_colums = []
    columns = ["timestamp"] + [f"motordata_{i}" for i in range(1, 9)]
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


    # magandmotor_df = pd.merge_asof(mag_df, motor_df, on="timestamp", direction="nearest")
    # merge_df = pd.merge_asof(magandmotor_df, motion_df, on="timestamp", direction="nearest")


    magandmotor_df = pd.merge_asof(mag_df, motor_df, on="timestamp", direction="nearest")
    print(magandmotor_df)
    print(motion_df)
    merge_df = pd.merge_asof(magandmotor_df, motion_df, on="timestamp", direction="nearest")
    merge_df = merge_df[final_colums]

    return merge_df.values.tolist()



magfile = myfunction.find_pickle_files("magsensor")
magsensor = myfunction.load_pickle(magfile)

motorfile = myfunction.find_pickle_files("motor")
motordata = myfunction.load_pickle(motorfile)

motionfile = myfunction.find_pickle_files("motioncapture")
motiondata = myfunction.load_pickle(motionfile)


  
magdata = []
for magrow in magsensor:
    magdata.append(mag_data_change2(magrow))


margedata = data_marge_v2(motiondata, motordata, magdata)



margedata.insert(0, ["time","rotate1","rotate2","rotate3","rotate4","force1","force2","force3","force4","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8","sensor9","Mc1x","Mc1y","Mc1z","Mc2x","Mc2y","Mc2z","Mc3x","Mc3y","Mc3z","Mc4x","Mc4y","Mc4z","Mc5x","Mc5y","Mc5z"])



df = pd.DataFrame(margedata[1:], columns=margedata[0])


original_len = len(df)


df = df[(df['Mc2x'] < 100000) & (df['Mc2y'] < 100000) & (df['Mc2z'] < 100000)]
df = df[(df['Mc3x'] < 100000) & (df['Mc3y'] < 100000) & (df['Mc3z'] < 100000)]
df = df[(df['Mc4x'] < 100000) & (df['Mc4y'] < 100000) & (df['Mc4z'] < 100000)]
df = df[(df['Mc5x'] < 100000) & (df['Mc5y'] < 100000) & (df['Mc5z'] < 100000)]

filtered_len = len(df)
removed_rows = original_len - filtered_len

print(f"motionで削除された行数: {removed_rows}")
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

mag_filtered_len = len(df)
removed_rows = filtered_len - mag_filtered_len

print(f"magで削除された行数: {removed_rows}")

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



filename = "0526_tubefinger_nohit_1500kai"

now = datetime.datetime.now()
filename = filename + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
# print(df.dtypes)
# print(type(df[2][10]))

df.to_pickle(filename)
