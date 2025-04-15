import serial
import csv
import time
import numpy as np
from datetime import datetime


class MagneticSensor:
    # def __init__(self):
    #     self.arduino = serial.Serial('COM6', 115200, timeout=1)
    #     self.datanum = 10  #センサーの数を変えるときはここ
    #     self.datas = []

    # #Arduinoから値を受け取って返り値で返す
    # def get_value(self):
    #     rowvalue = "///////////////////////////////////////////////////////////////////"

    #     while rowvalue.count("/") != self.datanum - 1:

    #         self.arduino.reset_input_buffer()  # 入力バッファをクリア
    #         self.arduino.reset_output_buffer()  # 出力バッファをクリア
    #         while self.arduino.in_waiting < 0:
    #             pass 
    #         rowvalue = self.arduino.readline().decode('utf-8', errors='ignore').rstrip()
    #     return rowvalue
    

    def __init__(self, port="COM4", baudrate=115200, timeout=1, datanum=10):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.datanum = datanum
        self.arduino = self._connect()

    def _connect(self):
        """
        Arduino デバイスに接続するための内部メソッド
        """
        while True:
            try:
                print(f"Trying to connect to {self.port}...")
                return serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            except serial.SerialException as e:
                print(f"Connection failed: {e}. Retrying in 2 seconds...")
                time.sleep(0.1)

    def get_value(self):
        """
        Arduino からデータを取得する。接続が切れた場合は再接続を試みる。
        """
        while True:
            try:
                # Arduinoのデータを取得する
                rowvalue = "///////////////////////////////////////////////////////////////////"
                while rowvalue.count("/") != self.datanum - 1:
                    self.arduino.reset_input_buffer()  # 入力バッファをクリア
                    self.arduino.reset_output_buffer()  # 出力バッファをクリア

                    while self.arduino.in_waiting <= 0:
                        pass

                    rowvalue = self.arduino.readline().decode('utf-8', errors='ignore').rstrip()
                return rowvalue

            except serial.SerialException as e:
                print(f"SerialException: {e}. Reconnecting...")
                self.arduino = self._connect()

            except Exception as e:
                print(f"Unexpected error: {e}. Retrying in 2 seconds...")
                time.sleep(0.1)
    
        
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


        
# mag = MagneticSensor()
# A = mag.get_value()
# print(A)