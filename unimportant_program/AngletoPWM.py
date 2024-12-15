#位置制御を行うが，トルクを測定し，そこまでみたいなのを試す

import sys, time
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting


# ID一覧eq
IDs = (ctypes.c_uint8 * 4)(1,2,3,4)

#---------------------------------------------
dev = dx2.DX2_OpenPort(setting.COMPort, setting.Baudrate)
if dev != None:
  # ID一覧分のDynamixelを検索しモデルp名を表示
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
  gPWMs = (ctypes.c_double * len(IDs))()
  nAngles = (ctypes.c_double * len(IDs))()
  gAngles = (ctypes.c_double * len(IDs))()
  
  dx2.DXL_GetPresentPWMs(dev, IDs, gPWMs, len(IDs))
  dx2.DXL_GetPresentAngles(dev, IDs, gAngles, len(IDs))

  changeRange = 5
  changeAngle = [0,0,0,0]

  while key != 'p':   # 'p'が押されると終了  
    if kb.kbhit():
      key = kb.getch()
      # ' '(スペース)を押す度にトルクイネーブルをトグル
      if key =='q':  
        gAngles[0] = gAngles[0]  +   changeRange
        changeAngle[0] = changeAngle[0] + changeRange
        print(changeAngle)
        dx2.DXL_SetGoalAngles (dev, IDs,  gAngles, len(IDs))
        print(dev)

      elif key == 'a':
        gAngles[0] = gAngles[0]  - changeRange
        changeAngle[0] = changeAngle[0] - changeRange
        print(changeAngle)
        dx2.DXL_SetGoalAngles (dev, IDs,  gAngles, len(IDs))

      elif key =='w':  
        gAngles[1] = gAngles[1]  + changeRange
        changeAngle[1] = changeAngle[1] + changeRange
        print(changeAngle)
        dx2.DXL_SetGoalAngles (dev, IDs,  gAngles, len(IDs))

      elif key == 's':
        gAngles[1] = gAngles[1]  - changeRange
        changeAngle[1] = changeAngle[1] - changeRange
        print(changeAngle)
        dx2.DXL_SetGoalAngles (dev, IDs,  gAngles, len(IDs))

      elif key =='e':  
        gAngles[2] = gAngles[2]  + changeRange
        changeAngle[2] = changeAngle[2] + changeRange
        print(changeAngle)
        dx2.DXL_SetGoalAngles (dev, IDs,  gAngles, len(IDs))

      elif key == 'd':
        gAngles[2] = gAngles[2]  - changeRange
        changeAngle[2] = changeAngle[2] - changeRange
        print(changeAngle)
        dx2.DXL_SetGoalAngles (dev, IDs,  gAngles, len(IDs))

      elif key =='r':  
        gAngles[3] = gAngles[3]  + changeRange
        changeAngle[3] = changeAngle[3] + changeRange
        print(changeAngle)
        dx2.DXL_SetGoalAngles (dev, IDs,  gAngles, len(IDs))

      elif key == 'f':
        gAngles[3] = gAngles[3]  - changeRange
        changeAngle[3] = changeAngle[3] - changeRange
        print(changeAngle)
        dx2.DXL_SetGoalAngles (dev, IDs,  gAngles, len(IDs))
      else:
        print
    # ID一覧分の角度を取得し表示ff
    if dx2.DXL_GetPresentPWMs(dev, IDs, nPWMs, len(IDs)):

      print('(', end='')
      print(('{:7.1f},'*len(nPWMs)).format(*nPWMs), end=')\r')
      sys.stdout.flush() 

  kb.set_normal_term()

  dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), False)

  dx2.DX2_ClosePort(dev)
else:
  print('Could not open COM port.')