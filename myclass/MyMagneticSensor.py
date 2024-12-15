import serial
import csv
import time
import numpy as np
from datetime import datetime


class MagneticSensor:
    def __init__(self):
        self.arduino = serial.Serial('COM6', 115200, timeout=1)
        self.datanum = 9  #センサーの数を変えるときはここ
        self.datas = []

    #Arduinoから値を受け取って返り値で返す
    def get_value(self):
        rowvalue = "///////////////////////////////////////////////////////////////////"
        while rowvalue.count("/") != self.datanum - 1:

            self.arduino.reset_input_buffer()  # 入力バッファをクリア
            self.arduino.reset_output_buffer()  # 出力バッファをクリア
            while self.arduino.in_waiting < 0:
                pass 
            rowvalue = self.arduino.readline().decode('utf-8', errors='ignore').rstrip()
        return rowvalue
        
    def change_data(self, rowvalue):
        resistance_value = np.empty(self.datanum)
        if rowvalue == '':
            return []
        split_value = np.array(rowvalue.split('/'))
        print(split_value)
        float_value = np.asarray(split_value, dtype=float)
        return float_value
    
    #クラス内にためる
    def store_data(self, data, nowtime = False):
        if nowtime == False:
            self.datas.append(data.tolist())
        else:
            data = data.tolist()
            data.insert(0, nowtime)
            self.datas.append(data)


        

