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

    print(mag_df, motor_df)
    # magandmotor_df = pd.merge_asof(mag_df, motor_df, on="timestamp", direction="nearest")
    # merge_df = pd.merge_asof(magandmotor_df, motion_df, on="timestamp", direction="nearest")


    motionandmotor_df = pd.merge_asof(motion_df, motor_df, on="timestamp", direction="nearest")
    merge_df = pd.merge_asof(motionandmotor_df, mag_df, on="timestamp", direction="nearest")
    merge_df = merge_df[final_colums]

    return merge_df.values.tolist()

with open('C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\magsensor20241205_005156.pickle', mode='br') as fi:
  magsensor = pickle.load(fi)

with open('C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\motor20241205_005156.pickle', mode='br') as fi:
  motordata = pickle.load(fi)

with open('C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\motioncapture20241205_005155.pickle', mode='br') as fi:
  motiondata = pickle.load(fi)
  
magdata = []
for magrow in magsensor:
    magdata.append(mag_data_change2(magrow))


print(magdata[0])





margedata = data_marge_v2(motiondata, motordata, magdata)


print(margedata[0])
margedata.insert(0, ["time","rotate1","rotate2","rotate3","rotate4","force1","force2","force3","force4","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8","sensor9","Mc1x","Mc1y","Mc1z","Mc2x","Mc2y","Mc2z","Mc3x","Mc3y","Mc3z","Mc4x","Mc4y","Mc4z","Mc5x","Mc5y","Mc5z"])



# myfunction.wirte_pkl( margedata, filename = "C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\marge_for_Mag\\margedata_formag")

