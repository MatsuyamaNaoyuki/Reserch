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


    def get_present_angles(self): #スタートとの角度の差を計算
        now_angles = (ctypes.c_double * len(self.IDs))()
        now_angles_list = []
        dx2.DXL_GetPresentAngles(self.dev, self.IDs, now_angles, len(self.IDs))
        if self.start_angles == None:
            raise SyntaxError("Do back to initial potiosn")
        for id in self.IDs:
            angle = now_angles[id-1] - self.start_angles[id-1]
            if id == 1 or id == 4:
                angle = angle * -1
            now_angles_list.append(angle)
        return now_angles_list
    
    def get_present_currents(self):
        nowforce = (ctypes.c_double)()
        nowforces = []
        for id in self.IDs:
            dx2.DXL_GetPresentCurrent(self.dev, id,  nowforce)
            if id == 1 or id == 4:
                nowforce.value = nowforce.value * -1
            nowforces.append(nowforce.value)
        return nowforces
    
    
    def move_to_points(self, angle_values ,times = 10):
        dx2.DXL_GetPresentAngles(self.dev, self.IDs, self.rotation_angles, len(self.IDs))
        gole_angles = self.rotation_angles
        setangle = []
        #方向の調整
        for i in range(len(angle_values)):
            id = i + 1
            if id ==  1 or id == 4:
                gole_angles[i] = self.start_angles[i] + angle_values[i] * -1

            else:
                gole_angles[i] = self.start_angles[i] + angle_values[i]
                
        # dx2.DXL_SetGoalAngles (self.dev, self.IDs,  gole_angles, len(self.IDs))
        dx2.DXL_SetGoalAnglesAndTime (self.dev, self.IDs, gole_angles, len(self.IDs),times)
        time.sleep(times)
        # dx2.DXL_SetGoalAnglesAndVelocities(self.dev, self.IDs, gole_angles, len(self.IDs))

        dx2.DXL_GetPresentAngles(self.dev, self.IDs, self.rotation_angles, len(self.IDs))

