#!/usr/bin/python3
# -*- coding: utf-8 -*-

# dx2libの追加APIを使用
# 1軸へ角度と遷移時間を同時指令
# DXL_SetGoalAngleAndTimeは移動距離と時間を指令するため、
# DXL_SetDriveModeでプロファイルを時間に設定しておく事。

import sys, time
from dx2lib import *  # dx2libをインポート

from setting import * # サンプル共通のポート・ボーレート・ID等

#---------------------------------------------
dev = DX2_OpenPort(COMPort, Baudrate)
if dev != None:
  # TargetID1のモデル情報を取得しモデル名を表示
  #   見つからなかった場合はプログラムを終了する
  if DXL_GetModelInfo(dev, TargetID1).contents.devtype != devtNONE:
    print (DXL_GetModelInfo(dev, TargetID1).contents.name.decode())

    # DynamixelをJointモード=3に変更
    DXL_SetOperatingMode(dev, TargetID1, 3)
    # プロファイルを時間(=4)に
    DXL_SetDriveMode(dev, TargetID1, 4)
    # トルクイネーブル
    DXL_SetTorqueEnable(dev, TargetID1, True)

    # 速度[deg]と遷移時間[sec]を順次指令
    print('1')
    DXL_SetGoalAngleAndTime2(dev, TargetID1,    0.0, 2.0)
    time.sleep(2)
    print('2')
    DXL_SetGoalAngleAndTime2(dev, TargetID1,  -90.0, 2.0)
    time.sleep(2)
    print('3')
    DXL_SetGoalAngleAndTime2(dev, TargetID1,    0.0, 2.0)
    time.sleep(2)
    print('4')
    DXL_SetGoalAngleAndTime2(dev, TargetID1,   90.0, 2.0)
    time.sleep(2)
    print('5')
    DXL_SetGoalAngleAndTime2(dev, TargetID1, -180.0, 2.0)
    time.sleep(2)
    print('6')
    DXL_SetGoalAngleAndTime2(dev, TargetID1,  180.0, 3.0)
    time.sleep(3)

    # トルクディスエーブル
    DXL_SetTorqueEnable(dev, TargetID1, False)
  else:
    print('Device information could not be acquired.')
  DX2_ClosePort(dev)
else:
  print('Could not open COM port.')
