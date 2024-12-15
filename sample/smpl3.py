#!/usr/bin/python3
# -*- coding: utf-8 -*-

# smpl2に引き続きコントロールテーブルのアイテムの名前を使って
# アクセスするクラス(dxl)を使用。
# 複数軸を操る場合は、その軸毎にdxlクラスをインスタンス化させ
# る必要がある。
# set/getはアクセスの仕方に多少柔軟性を持たせ、リストやタプル
# を使って複数のアイテムへのアクセスをまとめて記述できる。
# set_bulk/get_blukを使った複数軸を同時に扱う関数も扱うが、こ
# れらは複数のidを扱う都合から、クラス化されたコントロールテー
# ブルのコピーには反映されないことに注意。
# またget_bulkは取得できた情報のみを返すため、要求に満たない
# 場合の後処理にも注意が必要。

import kbhit, sys, time
from dataclasses import dataclass
from dx2lib import *
from dxl import *

from setting import * # サンプル共通のポート・ボーレート・ID等

# 対象IDのリスト
multi_ids = [1,2,3,4,5,6,7,8,9,10,11,12]

@dataclass
class PosInfo:
  gpos: int # goal position
  inc: int  # incremental
  max: int  # max position
  min: int  # min position

kb = kbhit.KBHit ()

dev = DX2_OpenPort(COMPort, Baudrate)
if dev != None:
  DX2_SetTimeOutOffset(dev, 50)

  # 正常にインスタンス化されたdxlのリスト
  dxs = []
  # 正常にインスタンス化されたidのリスト
  ids = ()
  # 各軸の現在位置・増分・最大/最小動作位置を保存するタプル
  posinfo = ()
  # 見つかったIDのデバイスを順次インスタンス化しdxsに追加
  for id in multi_ids:
    dxs += dxl(dev, id),
    # モデル名が設定されていれば正常
    if dxs[-1].modelname != '':
      print(f'{dxs[-1].id}:{dxs[-1].modelname}')
      # 1回の増分は動作範囲の1/32とした
      inc = (abs(dxs[-1].get('MaxPositionLimit') - dxs[-1].get('MinPositionLimit')) >> 5)
      posinfo += PosInfo(
        dxs[-1].get('PresentPosition'), # 現在位置
        inc,                            # 増分
        dxs[-1].get('MaxPositionLimit') - (inc << 3), # 最大値より少し小さめに設定
        dxs[-1].get('MinPositionLimit') + (inc << 3)  # 最小値より少し小さめに設定
      ),
      ids += id,
    else:
      dxs.pop(-1)
  print('ids=', ids)

  # 1軸以上見つかったとき
  if dxs:
    print('detect ids=', multi_ids)
    print('posinfo=', posinfo)

    # 初期条件を設定(トルクOFF->OPModeを4->トルクON->LED OFF)
    for dx in dxs:
      dx.set((('TorqueEnable', 0), ('OperatingMode', 4), ('TorqueEnable', 1), ('LED', 0)))

    # bulk用に現在位置の初期値を作成
    ppos=()
    for id in ids: ppos += (id, 'PresentPosition'),

    # 何かキーが押されるまで全軸へ角度指令
    while not kb.kbhit():
      # set_bulk用のタプル
      w = ()
      # 位置指令値を更新
      for i, dx in enumerate(dxs):
        posinfo[i].gpos += posinfo[i].inc
        if posinfo[i].gpos >= posinfo[i].max:
          posinfo[i].inc *= -1
          posinfo[i].gpos = posinfo[i].max
        elif posinfo[i].gpos <= posinfo[i].min:
          posinfo[i].inc *= -1
          posinfo[i].gpos = posinfo[i].min
        w += (dx.id, 'GoalPosition', posinfo[i].gpos),
      # bulk writeで位置を一括指令
      dxs[0].set_bulk(w)

      # LEDの反転とbulk readによる現在位置取得 (0.02*10=0.2秒の待ち)
      for n in range(10):
        for dx in dxs: dx.set('LED', dx.get('LED') ^ 1)
        prpos = dxs[0].get_bulk(ppos)
        for i, id in enumerate(ids): print('{:2}:{:6} '.format(id, prpos[i]), end='')
        print('', end='\r')
        time.sleep(0.02)

    del dxs

  DX2_ClosePort(dev)
else:
  print('Could not open COM port.')

kb.set_normal_term()
