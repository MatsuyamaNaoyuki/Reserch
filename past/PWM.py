
# dx2libの追加APIを使用
# 複数軸から現在角度取得

import sys, time
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting
# from dx2lib import *  # dx2libをインポート

# from setting import * # サンプル共通のポート・ボーレート・ID等

# ID一覧
IDs = (ctypes.c_uint8 * 4)(1,2,3,4)

#---------------------------------------------
dev = dx2.DX2_OpenPort(setting.COMPort, setting.Baudrate)
if dev != None:
  # ID一覧分のDynamixelを検索しモデル名を表示
  for id in IDs:
    print(id, dx2.DXL_GetModelInfo(dev,id).contents.name.decode())

  # ID一覧分のDynamixelをMultiTurnモード=4に変更
  dx2.DXL_SetOperatingModesEquival(dev, IDs, len(IDs), 16)
  # ID一覧分のDynamixelをトルクディスエーブル
  dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), False)

  # キー入力により処理を分岐

  key = ''
  kb = kbhit.KBHit()
  nPWMs = (ctypes.c_double * len(IDs))()
  gPWMs = (ctypes.c_double * len(IDs))()

  dx2.DXL_GetPresentPWMs(dev, IDs, gPWMs, len(IDs))


  while key != 'p':   # 'e'が押されると終了  
    if kb.kbhit():
      key = kb.getch()
      # ' '(スペース)を押す度にトルクイネーブルをトグル
      if key =='q':  
        dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)
        gPWMs[0] = gPWMs[0]  + 1
        dx2.DXL_SetGoalPWMs (dev, IDs,  gPWMs, len(IDs))

      elif key == 'a':
        dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)
        gPWMs[0] = gPWMs[0]  - 1
        dx2.DXL_SetGoalPWMs (dev, IDs,  gPWMs, len(IDs))
      elif key =='w':  
        dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)
        gPWMs[1] = gPWMs[1]  + 1
        dx2.DXL_SetGoalPWMs (dev, IDs,  gPWMs, len(IDs))

      elif key == 's':
        dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)
        gPWMs[1] = gPWMs[1]  - 1
        dx2.DXL_SetGoalPWMs (dev, IDs,  gPWMs, len(IDs))

      elif key =='e':  
        dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)
        gPWMs[2] = gPWMs[2]  + 1
        dx2.DXL_SetGoalPWMs (dev, IDs,  gPWMs, len(IDs))

      elif key == 'd':
        dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)
        gPWMs[2] = gPWMs[2]  - 1
        dx2.DXL_SetGoalPWMs (dev, IDs,  gPWMs, len(IDs))

      elif key =='r':  
        dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)
        gPWMs[3] = gPWMs[3]  + 1
        dx2.DXL_SetGoalPWMs (dev, IDs,  gPWMs, len(IDs))

      elif key == 'f':
        dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)
        gPWMs[3] = gPWMs[3]  - 1
        dx2.DXL_SetGoalPWMs (dev, IDs,  gPWMs, len(IDs))
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
 