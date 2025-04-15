#!/usr/bin/python3
# -*- coding: utf-8 -*-

# dx2lib.pyにはC言語で作られたAPIのラッパーが定義されている。
# 数値の扱いが異なるpythonとAPI間ではcyptesへの変換が生じる。
# なおソース中のアドレスはXM/XH/XDシリーズを前提としている事に注意。

import sys, time, ctypes
from unimportant_program.dx2lib import *  # dx2libをインポート

from setting import * # サンプル共通のポート・ボーレート・ID等

single_id = 1             # 1軸が対象の場合のID
multi_ids = 1,2,3,4,5,6,7 # 複数軸が対象の場合のID

dev = DX2_OpenPort(COMPort, Baudrate)
if dev != None:
  err = c_uint16(0)

  # 受信タイムアウト時間の延長
  DX2_SetTimeOutOffset(dev, 20)

  # ping
  print('>>DX2_Ping')
  for id in range(253):
    r = DX2_Ping(dev, id, err)
    print('\r ping = %d  err = %d     '%(id, err.value), end = '')
    if r: print('\n find', id)
  print()

  # boradcasting idを使ったping
  print('>>DX2_Ping2')
  num = c_int32(253)
  alm = (TDx2AlarmStatus * int(num.value))()
  r = DX2_Ping2(dev, num, alm, err)
  if int(num.value) > 0:
    for i in range(num.value):
      print(' find', alm[i].id, 'stat=', alm[i].Status)
  print(' err=', err.value)

  # LED(65)を点灯するインストラクションパケットを複数軸分送受信
  for id in multi_ids:
    print('>>DX2_TxPacket id=', id)
    inst = 0x03
    addr = 65
    onoff = 1
    txdat = (c_uint8 * 3)(addr & 0xff, (addr >> 8) & 0xff, onoff)
    DX2_TxPacket(dev, id, inst, txdat, len(txdat), err)
    print(' err=', err.value)

    print('>>DX2_RxPacket')
    rxdat = (ctypes.c_uint8 * 50)()
    rlen = c_uint32(0)
    timeout_ms = 200
    if DX2_RxPacket(dev, rxdat, len(rxdat), rlen, timeout_ms, err):
      print(' rxdat=', ('{:02X} ' * rlen.value).format(*rxdat))
    print(' err=', err.value)

  # Indirect Data 1(224)をRAM代わりに読み書き
  for i in range(10):
    print('>>DX2_WriteByteData id=', single_id)
    addr = 224
    wbdat = c_uint8(i)
    DX2_WriteByteData(dev, single_id, addr, wbdat, err)
    print(' err=', err.value)

    print('>>DX2_ReadByteData id=', single_id)
    addr = 224
    rbdat = c_uint8(0)
    if DX2_ReadByteData(dev, single_id, addr, rbdat, err):
      print (' rbdat=', rbdat.value)
    print(' err=', err.value)

  for i in range(10):
    print('>>DX2_WriteWordData id=', single_id)
    addr = 224
    wwdat = c_uint16(i * 256)
    DX2_WriteWordData(dev, single_id, addr, wwdat, err)
    print(' err=', err.value)

    print('>>DX2_ReadWordData id=', single_id)
    addr = 224
    rwdat = c_uint16(0)
    if DX2_ReadWordData(dev, single_id, addr, rwdat, err):
      print (' rwdat=', rwdat.value)
    print(' err=', err.value)

  for i in range(10):
    print('>>DX2_WriteWordData id=', single_id)
    addr = 224
    wldat = c_uint32(i * 65536)
    DX2_WriteLongData(dev, single_id, addr, wldat, err)
    print(' err=', err.value)

    print('>>DX2_ReadWordData id=', single_id)
    addr = 224
    rldat = c_uint32(0)
    if DX2_ReadLongData(dev, single_id, addr, rldat, err):
      print (' rldat=', rldat.value)
    print(' err=', err.value)

  # 先頭から147バイト読み出し
  print('>>DX2_ReadBlockData')
  addr = 0
  rbary = (c_uint8 * 147)()
  if DX2_ReadBlockData(dev, single_id, addr, rbary, len(rbary), err):
    print(' rbary=', ('{:02X} ' * len(rbary)).format(*rbary))
  print(' err=', err.value)

  # LEDを消灯するBULKデータを複数軸分送信
  print('>>DX2_WriteBulkData')
  wbulkdat = []
  addr = 65
  wlen = 1
  onoff = 0
  for id in multi_ids:
    wbulkdat += [id, addr & 0xff, (addr >> 8) & 0xff, wlen & 0xff, (wlen >> 8) & 0xff, onoff]
  wbary = (c_uint8 * len(wbulkdat))(*wbulkdat)
  print(' wbary=', ('{:02X} ' * len(wbary)).format(*wbary))
  DX2_WriteBulkData(dev, wbary, len(wbary), err)
  print(' err=', err.value)

  DX2_ClosePort(dev)
else:
  print('Could not open COM port.')
