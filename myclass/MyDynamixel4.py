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

    def move(self, id, angle_displacement):
        idi = id - 1
        #現在の角度（内部）を取得
        dx2.DXL_GetPresentAngles(self.dev, self.IDs, self.rotation_angles, len(self.IDs))
        gole_angles = self.rotation_angles

        #方向の調整
        if id ==  1 or id == 4:
            setangle = angle_displacement * -1
        else:
            setangle = angle_displacement

        gole_angles[idi] = gole_angles[idi] + setangle
        #実際に移動（内部角度を入力）
        dx2.DXL_SetGoalAngles (self.dev, self.IDs,  gole_angles, len(self.IDs))
        dx2.DXL_GetPresentAngles(self.dev, self.IDs, self.rotation_angles, len(self.IDs))
        
        
        
    def manual_move(self):#そのうち抽象化したいね
        key = ''
        kb = kbhit.KBHit()
        while key != 'p':   # 'p'が押されると終了  
            if kb.kbhit():
                key = kb.getch()
        # ' '(スペース)を押す度にトルクイネーブルをトグル
                if key =='q':  
                    self.move(1, 10)
                if key =='w':  
                    self.move(2, 10)
                if key =='e':  
                    self.move(3, 10)
                if key =='r':  
                    self.move(4, 10)
                if key =='a':  
                    self.move(1, -10)
                if key =='s':  
                    self.move(2, -10)
                if key =='d':  
                    self.move(3, -10)
                if key =='f':  
                    self.move(4, -10)
                if key =='t':  
                    self.move(1, 100)
                if key =='y':  
                    self.move(2, 100)
                if key =='u':  
                    self.move(3, 100)
                if key =='i':  
                    self.move(4, 100)
                if key =='g':  
                    self.move(1, -100)
                if key =='h':  
                    self.move(2, -100)
                if key =='j':  
                    self.move(3, -100)
                if key =='k':  
                    self.move(4, -100)
                print(self.get_present_angles())