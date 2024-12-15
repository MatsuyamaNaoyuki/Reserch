#トルクを入力し位置で制御する

import sys, time
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting
import time

def change_torque(dev, IDs, id, new_torque):
    for i in range(len(IDs)):
        if IDs[i] == id:
            id_subscript = i
    now_torque = (ctypes.c_double)()
    now_angle = (ctypes.c_double)()
    dx2.DXL_GetPresentPWM(dev, id, now_torque)
    count = 0
    while(abs(new_torque - abs(now_torque.value)) > 0.25):
        dx2.DXL_GetPresentAngle(dev, id, now_angle)
        if new_torque - abs(now_torque.value) > 0:
            if id == 1 or 4:
                new_angle = now_angle.value - 2 
            else:
                new_angle = now_angle.value + 2
        else:
            if id == 1 or 4:
                new_angle = now_angle.value + 2 
            else:
                new_angle = now_angle.value - 2
        print("now_angle =", now_angle.value)
        print("new_angle =", new_angle)

        dx2.DXL_SetGoalAngle(dev, id, new_angle)
        time.sleep(0.2)
        dx2.DXL_GetPresentPWM(dev, id, now_torque)
        count = count + 1
        if count > 200:
           print("over")
           break
        


          
    

# ID一覧
IDs = (ctypes.c_uint8 * 4)(1,2,3,4)

#---------------------------------------------
dev = dx2.DX2_OpenPort(setting.COMPort, setting.Baudrate)
if dev != None:
  # ID一覧分のDynamixelを検索しモデル名を表示
  for id in IDs:
    print(id, dx2.DXL_GetModelInfo(dev,id).contents.name.decode())

  # ID一覧分のDynamixelをMultiTurnモード=4に変更
  dx2.DXL_SetOperatingModesEquival(dev, IDs, len(IDs), 4)
  # ID一覧分のDynamixelをトルクディスエーブル
  dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)

  # キー入力により処理を分岐

  key = ''
  kb = kbhit.KBHit()
  nPWMs = (ctypes.c_double * len(IDs))()
  gPWMs = (ctypes.c_double * len(IDs))(0,0,0,0)
  nAngles = (ctypes.c_double * len(IDs))()
  gAngles = (ctypes.c_double * len(IDs))()
  

  dx2.DXL_GetPresentAngles(dev, IDs, gAngles, len(IDs))

  change_range = 3

  while key != 'p':   # 'p'が押されると終了  
    if kb.kbhit():
      key = kb.getch()
      # ' '(スペース)を押す度にトルクイネーブルをトグル
      if key =='q':  
        id = 1
        gPWMs[0] = gPWMs[0]  + change_range
        print("gPWMs[0] = ", gPWMs[0])
        change_torque(dev, IDs, id, gPWMs[0])

      elif key == 'a':
        id = 1
        gPWMs[0] = gPWMs[0]  - change_range
        print("gPWMs[0] = ", gPWMs[0])
        change_torque(dev, IDs, id, gPWMs[0])

      elif key =='w': 
        id  = 2 
        gPWMs[1] = gPWMs[1]  + change_range
        print("gPWMs[1] = ", gPWMs[1])
        change_torque(dev, IDs, id, gPWMs[1])

      elif key == 's':
        id  = 2
        gPWMs[1] = gPWMs[1]  - change_range
        print("gPWMs[1] = ", gPWMs[1])
        change_torque(dev, IDs, id, gPWMs[1])

      elif key =='e': 
        id  = 3 
        gPWMs[2] = gPWMs[2]  + change_range
        print("gPWMs[2] = ", gPWMs[2])
        change_torque(dev, IDs, id, gPWMs[2])

      elif key == 'd':
        id = 3
        gPWMs[2] = gPWMs[2]  - change_range
        print("gPWMs[2] = ", gPWMs[2])
        change_torque(dev, IDs, id, gPWMs[2])

      elif key =='r':
        id = 4  
        gPWMs[3] = gPWMs[3]  + change_range
        print("gPWMs[3] = ", gPWMs[3])
        change_torque(dev, IDs, id, gPWMs[3])

      elif key == 'f':
        id = 4
        gPWMs[3] = gPWMs[3]  - change_range
        print("gPWMs[3] = ", gPWMs[3])
        change_torque(dev, IDs, id, gPWMs[3])
      else:
        print
    # ID一覧分の角度を取得し表示
    if dx2.DXL_GetPresentPWMs(dev, IDs, nPWMs, len(IDs)):
      print('(', end='')
      print(('{:7.1f},'*len(nPWMs)).format(*nPWMs), end=')\r')
      sys.stdout.flush() 
  kb.set_normal_term()

  dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), False)

  dx2.DX2_ClosePort(dev)
else:
  print('Could not open COM port.')



 