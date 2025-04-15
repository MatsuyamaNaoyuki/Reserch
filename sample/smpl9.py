#!/usr/bin/python3
# -*- coding: utf-8 -*-

# dx2libの追加APIを使用
# 2軸へ角速度指令

import sys, time
from unimportant_program.dx2lib import *  # dx2libをインポート

from setting import * # サンプル共通のポート・ボーレート・ID等

#---------------------------------------------
# 角速度指令と時間待ち関数
#---------------------------------------------
def SetVelo(dev, ids, velos, sec):
  for velo in velos: print(velo, ' ', end='')
  print()
  # 軸分の角速度を指令
  DXL_SetGoalVelocities(dev, ids, (c_double * len(ids))(*velos), len(ids))
  # 指定時間待つ
  time.sleep(sec)

#---------------------------------------------
dev = DX2_OpenPort(COMPort, Baudrate)
if dev != None:
  # IDの一覧
  IDs = (c_uint8 * 2)(TargetID1, TargetID2)
  # 指定IDのモデル情報を取得しモデル名を表示
  #   内部DBへの登録が目的だが、指定されたIDが見つからなかった場合は
  #   以後の操作でそのIDは通信対象から外れる
  for id in IDs:
    print(id, DXL_GetModelInfo(dev,id).contents.name.decode())

  # DynamixelをVelocityモード=1に変更
  DXL_SetOperatingModesEquival(dev, IDs, 2, 1)

  DXL_SetTorqueEnablesEquival(dev, IDs, 2, True)

  # 2軸分の角速度[deg/sec]と時間待ちを順次指令
  SetVelo(dev, IDs, [ 150, -150], 3);
  SetVelo(dev, IDs, [-150,  150], 3);
  SetVelo(dev, IDs, [ 150,  150], 3);
  SetVelo(dev, IDs, [-150, -150], 3);
  SetVelo(dev, IDs, [   0,    0], 1);

  DXL_SetTorqueEnablesEquival(dev, IDs, 2, False)

  DX2_ClosePort(dev)
else:
  print('Could not open COM port.')
