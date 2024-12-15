

# dx2libの追加APIを使用
# 複数軸から現在角度取得

import sys, time, kbhit
from dx2lib import *  # dx2libをインポート

from setting import * # サンプル共通のポート・ボーレート・ID等

# ID一覧
IDs = (c_uint8 * 8)(1,2,3,4,5,6,7,8)

#---------------------------------------------
dev = DX2_OpenPort(COMPort, Baudrate)
if dev != None:
  # ID一覧分のDynamixelを検索しモデル名を表示
  for id in IDs:
    print(id, DXL_GetModelInfo(dev,id).contents.name.decode())

  # ID一覧分のDynamixelをMultiTurnモード=4に変更
  DXL_SetOperatingModesEquival(dev, IDs, len(IDs), 4)
  # ID一覧分のDynamixelをトルクディスエーブル
  DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), False)

  # キー入力により処理を分岐
  ten = False
  k = ''
  kb = kbhit.KBHit()
  pangles = (c_double * len(IDs))()
  while k != 'e':   # 'e'が押されると終了
    if kb.kbhit():
      k = kb.getch()
      # ' '(スペース)を押す度にトルクイネーブルをトグル
      if k==' ':
        ten = not ten
        DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), ten)
        print('\nTorque Enable='),
        print(ten)
      else:
        print
    # ID一覧分の角度を取得し表示
    if DXL_GetPresentAngles(dev, IDs, pangles, len(IDs)):
      print('(', end='')
      print(('{:7.1f},'*len(pangles)).format(*pangles), end=')\r')
      sys.stdout.flush()
  kb.set_normal_term()

  DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), False)

  DX2_ClosePort(dev)
else:
  print('Could not open COM port.')
