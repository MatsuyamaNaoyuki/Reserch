import serial
import csv
import time
import numpy as np
from datetime import datetime


class BendingSensor:
    def __init__(self):
        self.arduino = serial.Serial('COM3', 115200, timeout=1)
        self.datanum = 3
        self.datas = []

    def get_value(self):
        rowvalue = "a"
        while rowvalue.count("/") != self.datanum - 1:
            self.arduino.reset_input_buffer()  # 入力バッファをクリア
            self.arduino.reset_output_buffer()  # 出力バッファをクリア
            while self.arduino.in_waiting < 0:
                pass 
            rowvalue = self.arduino.readline().decode('utf-8', errors='ignore').rstrip()
        return rowvalue
        
    
    def change_data(self, rowvalue, change_resistace = False):
        resistance_value = np.empty(8)
        split_value = np.array(rowvalue.split('/'))
        float_value = np.asarray(split_value, dtype=float)


        Vcc = 5.0
        Rt = 300000
        for i in range(len(float_value)):
            Vx = float_value[i] * Vcc / 1024
            resistance_value[i] = Rt / (Vcc - Vx) * Vx
            
        if change_resistace == True:
            return resistance_value
        else:
            return float_value
        
    def store_data(self, data):
        self.datas.append(data.tolist())

        





# bending1 = BendingSensor()
# value = bending1.get_value()
# floatvalue = bending1.change_data(value)
# bending1.store_data(floatvalue)
# value = bending1.get_value()
# floatvalue = bending1.change_data(value)
# bending1.store_data(floatvalue)
# print(bending1.datas)
