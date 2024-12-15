#!/usr/bin/python3
# -*- coding: utf-8 -*-

# dx2libの追加APIを使用
# 1軸のLED明滅しか行っていないが、追加APIの最低限の使い方を
# 紹介する。
# 追加APIはデバイスの違いを吸収する目的で作られており、各制
# 御値には物理値を扱うようになっている。
# 先のアイテム名でアクセスするよりも更にデバイスの素性を隠蔽
# して運用する事が目的である。
# しかしながらこちらは低位APIと同様に変数はctypesが前提であ
# ることに注意が必要。

import sys, time
from ctypes import *
from dx2lib import *  # dx2libをインポート

from setting import * # サンプル共通のポート・ボーレート・ID等

# DXL_PrintDevicesListにて使用
if ('mingw' in get_platform()) or ('win' in get_platform()):
  libc = cdll.msvcrt
else:
  libc = cdll.LoadLibrary("libc.so.6")

# DXL_ScanDevicesの結果を保存する配列
IDs = (c_uint8 * 253)()

#---------------------------------------------
# ポートを開いてTDeviceIDを取得
dev = DX2_OpenPort(COMPort, Baudrate)
if dev != None:
  # ひとまず全ID分を検索
  #   DXL_ScanDevicesの検索は全IDが対象なので完了まで時間がかかる
  #   内部DBへの登録が目的だが、指定されたIDが見つからなかった場合は
  #   以後の操作でそのIDは通信対象から外れる
  num = DXL_ScanDevices(dev, IDs)
  print ('detect num=', num)
  # 取得されたモデル情報の一覧を表示
  DXL_PrintDevicesList (libc.printf)

  # 1個以上見つかった
  if num > 0:
    # LED点灯
    for i in range(num): DXL_SetLED(dev, IDs[i], True)
    # 1秒待ち
    time.sleep(1)
    # LED消灯
    for i in range(num): DXL_SetLED(dev, IDs[i], False)
  else:
    print('Device information could not be acquired.')
  # ポートを閉じる(必須)
  DX2_ClosePort(dev)
else:
  print('Could not open COM port.')
