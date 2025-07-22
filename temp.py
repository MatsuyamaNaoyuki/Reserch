import sys,os, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting
import csv
import pprint

class MyDynamixel():
    def __init__(self):
        
        self.IDs = (ctypes.c_uint8 * 255)()
        self.dev = dx2.DX2_OpenPort(setting.COMPort, setting.Baudrate)
        dx2.DXL_ScanDevices(self.dev, self.IDs)
        non_zero_ids = [x for x in self.IDs if x != 0]
        ArrayType = ctypes.c_uint8 * len(non_zero_ids)
        self.IDs= ArrayType(*non_zero_ids)

        if self.dev != None:
            # ID一覧分のDynamixelを検索しモデルp名を表示
            for id in self.IDs:
                print(id, dx2.DXL_GetModelInfo(self.dev,id).contents.name.decode())
        else:
            print('Could not open COM port.')
        print("raw list:", list(self.IDs)) 
        
        #ID一覧分のDynamixelをMultiTurnモード=4に変更
        dx2.DXL_SetOperatingModesEquival(self.dev, self.IDs, len(self.IDs), 4)
        # ID一覧分のDynamixelをトルクディスエーブル
        dx2.DXL_SetTorqueEnablesEquival(self.dev, self.IDs, len(self.IDs), True)
        self.rotation_angles = (ctypes.c_double * len(self.IDs))()
        self.start_angles = (ctypes.c_double * len(self.IDs))()
        self.force = (ctypes.c_double * len(self.IDs))()
        dx2.DXL_GetPresentAngles(self.dev, self.IDs, self.start_angles, len(self.IDs))
        self.anglerecord = []
        
        
Motor = MyDynamixel()