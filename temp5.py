import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from myclass.MyMagneticSensor import MagneticSensor
import pandas as pd


def mag_data_change(row):
    split_value = row.split('/')
    if len(split_value) != 9:
        split_value = split_value[1:]
    int_list = [int(c) for c in split_value] 
    return int_list



Ms = MagneticSensor()
columns = ['sensor1', 'sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9']
sensordf = pd.DataFrame(columns=columns)





for i in range(10):
    magdata = Ms.get_value()
    row = mag_data_change(magdata)
    sensordf.loc[len(sensordf)] = row
    
    
    
    # (df.loc['row_1':'row_3', ['col_2', 'col_0']])
    
print(sensordf.loc[:5, ["sensor1"]])




