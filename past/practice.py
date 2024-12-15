
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
IDs = (ctypes.c_uint8 * 2)(1,3)

#---------------------------------------------
dev = dx2.DX2_OpenPort(setting.COMPort, setting.Baudrate)
if dev != None:
  # ID一覧分のDynamixelを検索しモデル名を表示
  for id in IDs:
    print(id, dx2.DXL_GetModelInfo(dev,id).contents.name.decode())

  # ID一覧分のDynamixelをMultiTurnモード=4に変更
  dx2.DXL_SetOperatingModesEquival(dev, IDs, len(IDs), 4)
  # ID一覧分のDynamixelをトルクディスエーブル
  dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), False)

  # キー入力により処理を分岐

  key = ''
  kb = kbhit.KBHit()
  pangles = (ctypes.c_double * len(IDs))()
  gangles = (ctypes.c_double * len(IDs))()

  dx2.DXL_GetPresentAngles(dev, IDs, gangles, len(IDs))


  while key != 'e':   # 'e'が押されると終了  
    if kb.kbhit():
      key = kb.getch()
      # ' '(スペース)を押す度にトルクイネーブルをトグル
      if key ==' ':  
        dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)
        gangles[1] = gangles[1]  + 10
        dx2.DXL_SetGoalAngles (dev, IDs,  gangles, len(IDs))
        print('\nTorque Enable='),

      else:
        print
    # ID一覧分の角度を取得し表示
    if dx2.DXL_GetPresentAngles(dev, IDs, pangles, len(IDs)):
      print('(', end='')
      print(('{:7.1f},'*len(pangles)).format(*pangles), end=')\r')
      sys.stdout.flush()
  kb.set_normal_term()

  dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), False)

  dx2.DX2_ClosePort(dev)
else:
  print('Could not open COM port.')
 