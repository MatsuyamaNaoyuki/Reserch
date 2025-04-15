#!/usr/bin/python3
# -*- coding: utf-8 -*-

# dx2libの追加APIを使用
# 1軸のLED明滅

import sys, time
from unimportant_program.dx2lib import *  # dx2libをインポート

from setting import * # サンプル共通のポート・ボーレート・ID等

#---------------------------------------------
dev = DX2_OpenPort(COMPort, Baudrate)
if dev != None:
  # 指定IDのモデル情報を取得
  #   DXL_ScanDevicesの検索は全IDが対象なので時間がかかるが、
  #   DXL_GetModelInfoでIDを指定して見つかれば即時登録される
  #   内部DBへの登録が目的だが、指定されたIDが見つからなかった場合は
  #   以後の操作でそのIDは通信対象から外れる
  if DXL_GetModelInfo(dev, TargetID1).contents.devtype != devtNONE:
    # 取得されたモデル情報からモデル名を表示
    print (DXL_GetModelInfo(dev, TargetID1).contents.name.decode())

    # LED点灯
    DXL_SetLED(dev, TargetID1, True)
    # 1秒待ち
    time.sleep(1)
    # LED消灯
    DXL_SetLED(dev, TargetID1, False)
  else:
    print('Device information could not be acquired.')
  DX2_ClosePort(dev)
else:
  print('Could not open COM port.')
