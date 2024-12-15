#!/usr/bin/python3
# -*- coding: utf-8 -*-

# dx2libの低位のAPIは使用するDynamixelのコントロールテーブルの一覧
# とアドレスを見ながらのコーディングが必須である。
# ここではコントロールテーブルのアイテムの名前を使ってアクセスする
# クラス(dxl)を紹介する。
# このクラスは1個のDynamixelをクラス化し、コントロールテーブル上の
# アイテム名をset/get関数の引数に与えてアクセスする。
# アイテム名はe-manual上の命名を元にパスカルケースで決めているが、
# 同類の機能を持ったアイテムでもモデルによって異なる場合もあるため、
# モデル違いまでを吸収するものではない。
# なおget関数はdxlのインスタンス化時にDynamixelのコントロールテーブ
# ルを読み出した内容を返し、デフォルトでは通信を行わない。もし現在
# の情報を通信した上で取得する場合は、update=Trueを指定する。

import kbhit, sys, time
from dx2lib import *  # dx2libをインポート
from dxl import *     # dxlをインポート

from setting import * # サンプル共通のポート・ボーレート・ID等

kb = kbhit.KBHit ()

# ポートをオープン
dev = DX2_OpenPort(COMPort, Baudrate)
if dev != None:
  DX2_SetTimeOutOffset(dev, 20)

  # TargetID1を制御対象にしてdxlクラスをインスタンス化
  dx = dxl(dev, TargetID1)
  # modelnameが空でなければサポートしているDynamixel
  if dx.modelname != '':
    print ('ID=%d modelname=%s'%(dx.id, dx.modelname))

    # コントロールテーブルに含まれる全アイテムとデータを
    # アドレスを元に取得し表示してみる
    for addr in range(1000):
      n = dx.itemname(addr)
      if n: print(f'{n}({addr})={dx.get(n)}, ', end='')
    else:
      print()

    print('Set TorqueEnable=', dx.set('TorqueEnable', 1))
    print('Get TorqueEnable=', dx.get('TorqueEnable'))
    print('Set IndirectData1_28=', dx.set('IndirectData1_28', (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32)))
    print('Get IndirectData1_28=', dx.get('IndirectData1_28'))
    print('Set multiple=', dx.get(('hogehoge', 'IndirectAddress1_28','IndirectData1_28','IndirectAddress29_56','IndirectData29_56'), True))

    # LED点滅
    for i in range(20):
      led = dx.get('LED') ^ 1
      print('led=', led, end = '    \r', flush = True)
      dx.set('LED', led)
      time.sleep(0.2)
    print()

    # 制御OFF
    dx.set('TorqueEnable', 0)
    # 動作モードを位置制御に設定
    dx.set('OperatingMode', 3)
    # 制御ON
    dx.set('TorqueEnable', 1)

    # 動作範囲を取得
    maxpos = dx.get('MaxPositionLimit')
    print('maxposlimit=', maxpos)
    minpos = dx.get('MinPositionLimit')
    print('minposlimit=', minpos)

    # 動作範囲を元に角度指令
    CtrlRange= (
      (int((maxpos + minpos) / 2), maxpos, int(abs(maxpos - minpos) / 400)),
      (maxpos, minpos, -int(abs(maxpos - minpos) / 400)),
      (minpos, int((maxpos + minpos) / 2), int(abs(maxpos - minpos) / 400))
    )
    for r in CtrlRange:
      for p in range(*r):
        print('gpos=', p, end = '   \r', flush = True)
        dx.set('GoalPosition', p)
        time.sleep(0.01)
    else:
      print()

    # 制御OFF
    dx.set('TorqueEnable', 0)

  else:
    print('Communication failed or unsupported device.')

  DX2_ClosePort(dev)
else:
  print('Could not open COM port.')

kb.set_normal_term()
