#!python3
# -*- coding: utf-8 -*-

# dx2libの追加APIを使用
# 1軸へ角度と角速度を同時指令

import sys, time
from dx2lib import *  # dx2libをインポート

from setting import * # サンプル共通のポート・ボーレート・ID等

#---------------------------------------------
dev = DX2_OpenPort(COMPort, Baudrate)
if dev != None:
  # 指定IDのモデル情報を取得しモデル名を表示
  #   ここでは見つからなかった場合はプログラムを終了する
  if DXL_GetModelInfo(dev, TargetID).contents.devtype != devtNONE:
    print (DXL_GetModelInfo(dev, TargetID).contents.name.decode())

    # DynamixelをJointモード=3に変更
    DXL_SetOperatingMode(dev, TargetID, 3)
    # トルクイネーブル
    DXL_SetTorqueEnable(dev, TargetID, True)

    # 角度[deg]と角速度を[deg/sec]を順次指令
    DXL_SetGoalAngleAndVelocity(dev, TargetID,  90.0, 90.0)
    time.sleep(1.5)
    DXL_SetGoalAngleAndVelocity(dev, TargetID,   0.0, 90.0)
    time.sleep(1.5)
    DXL_SetGoalAngleAndVelocity(dev, TargetID, -90.0, 90.0)
    time.sleep(1.5)
    DXL_SetGoalAngleAndVelocity(dev, TargetID,   0.0, 90.0)
    time.sleep(2)

    # トルクディスエーブル
    DXL_SetTorqueEnable(dev, TargetID, False)

  else:
    print('Device information could not be acquired.')
  DX2_ClosePort(dev)
else:
  print('Could not open COM port.')
