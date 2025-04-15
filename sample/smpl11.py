#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 簡易的なモーション再生を行うクラス(motion)を使用。
# できるだけpythonの変数やlist、tupleを使用してctypeの使用を軽減。
# キャプチャしたデータのファイル化も行える。

import kbhit, time, threading, os, json
from unimportant_program.dx2lib import *  # dx2libをインポート
from motion import *  # 

from setting import * # サンプル共通のポート・ボーレート・ID等

# モーションデータファイル
motionfile = 'm.txt'

# スレッド終了用フラグ
term = False;

# ID一覧
IDs1 = 1,2,3,4
IDs2 = 5,6,7,8
IDs8 = 1,2,3,4,5,6,7,8

# dxlib用にtupleをctypesへ予め変換しておく
c_ids = (c_uint8 * len(IDs8))(*IDs8)

# モーションキャプチャ用リスト
mym = []

# モーションデータ
#   Poseをtuple化
m_0 = (Pose)((   0,   0,   0,   0,   0,   0,   0,   0), 1.0),

m_1 = ((Pose)((  90,  90,  90,  90), 0.5),
       (Pose)((   0,   0,   0,   0), 0.5),
       (Pose)((  90,  90,  90,  90), 0.5),
       (Pose)((   0,   0,   0,   0), 0.5),
       (Pose)((  90,  90,  90,  90), 0.5),
       (Pose)((   0,   0,   0,   0), 0.5),
       (Pose)((  90,  90,  90,  90), 0.5),
       (Pose)((   0,   0,   0,   0), 0.5),
       (Pose)((  90,  90,  90,  90), 0.5),
       (Pose)((   0,   0,   0,   0), 0.5))

m_2 = ((Pose)(( -90, -90, -90, -90), 1.0),
       (Pose)((   0,   0,   0,   0), 0.5),
       (Pose)(( -90, -90, -90, -90), 1.0),
       (Pose)((   0,   0,   0,   0), 0.5),
       (Pose)(( -90, -90, -90, -90), 1.0),
       (Pose)((   0,   0,   0,   0), 0.5),
       (Pose)(( -90, -90, -90, -90), 1.0),
       (Pose)((   0,   0,   0,   0), 0.5))

m_3 = (Pose)(( 180, 180, 180, 180, 180, 180, 180, 180), 1.0),

m_4 = (Pose)((-180,-180,-180,-180,-180,-180,-180,-180), 1.0),

m_5 = ((Pose)(( -30, -30, -30, -30, -30, -30, -30, -30), 0.2),
       (Pose)((  30,  30,  30,  30,  30,  30,  30,  30), 0.2),
       (Pose)(( -30, -30, -30, -30, -30, -30, -30, -30), 0.2),
       (Pose)((  30,  30,  30,  30,  30,  30,  30,  30), 0.2),
       (Pose)(( -30, -30, -30, -30, -30, -30, -30, -30), 0.2),
       (Pose)((  30,  30,  30,  30,  30,  30,  30,  30), 0.2),
       (Pose)(( -30, -30, -30, -30, -30, -30, -30, -30), 0.2),
       (Pose)((  30,  30,  30,  30,  30,  30,  30,  30), 0.2))

m_6 = ((Pose)(( -45,  45, -45,  45, -45,  45, -45,  45), 2.0),
       (Pose)((  90,  90,  90,  90,  90,  90,  90,  90), 2.0),
       (Pose)(( -30, -30, -30, -30, -30, -30, -30, -30), 2.0),
       (Pose)((-180,-180,-180,-180,-180,-180,-180,-180), 2.0))

#---------------------------------------------
# モニタスレッド
#---------------------------------------------
def Mon():
  v = (TAngleVelocity * len(IDs8))()
  prevang = (c_double * len(IDs8))()
  while not term:
    if DXL_GetPresentAngles(dev, c_ids, prevang, len(c_ids)):
      print('angle =',('{:7.1f} '*len(prevang)).format(*prevang), end='     \r')
    else:
      print('\nNG')
    time.sleep(0.005)

#---------------------------------------------
dev = DX2_OpenPort(COMPort, Baudrate)
if dev != None:
  DX2_SetTimeOutOffset(dev, 20)

  # モーションクラスをインスタンス化
  mt = motion(dev)
  # ID一覧分のDynamixelを検索しモデル名を表示
  for id in IDs8:
    print(id,DXL_GetModelInfo(dev,id).contents.name.decode())
  # 制御モードを拡張位置制御に
  DXL_SetOperatingModesEquival(dev, c_ids, len(c_ids), 4)
  # ドライブモードを時間プロファイルに(必須)
  DXL_SetDriveModesEquival (dev, c_ids, len(c_ids), 0x4)

  # 現在角度を取得するスレッドを開始
  thread_1 = threading.Thread(target=Mon)
  thread_1.start()

  kb = kbhit.KBHit ()
  k = 'a'
  # キーボードからの数値入力で再生するモーションを選択
  while k != 'q':
    k = kb.getch()
    print('\n'+k)
    # トルクON
    if   k == 'o': mt.enablectrl(IDs8, True)
    # トルクOFF
    elif k == 'f': mt.enablectrl(IDs8, False)
    # 各モーションを再生
    elif k == '0': mt.start(0, IDs8, m_0)
    elif k == '1': mt.start(1, IDs1, m_1)
    elif k == '2': mt.start(2, IDs2, m_2)
    elif k == '3': mt.start(0, IDs8, m_3)
    elif k == '4': mt.start(0, IDs8, m_4)
    elif k == '5': mt.start(0, IDs8, m_5)
    elif k == '6': mt.start(0, IDs8, m_6)
    # 全モーション停止
    elif k == 'e':
      for i in range(4):mt.stop(i)
    # モーションをキャプチャ
    elif k == 'm':
      v = (TAngleVelocity * len(IDs8))()
      prevang = (c_double * len(IDs8))()
      ind = 0;
      mym[:] = []
      for i in range(4):mt.stop(i)
      mt.enablectrl(IDs8, False)
      prev_time = time.time()
      while not kb.kbhit():
        start_time = time.time()
        if DXL_GetPresentAngles(dev, c_ids, prevang, len(c_ids)):
          ca = []
          for ang in prevang: ca.append(round(ang, 1))
          print('angle =',('{:7.1f} '*len(ca)).format(*ca), end='     \n')
          mym.append(Pose(ca, time.time() - prev_time))
        ind += 1
        prev_time = start_time
        time.sleep(0.05)
      kb.getch()

    # キャプチャしたモーションを再生
    elif k == 'p': mt.start(0, IDs8, mym)

    # モーションの書き出し
    elif k == 's':
      print('save motion file')
      if len(mym) > 0:
        with open(motionfile,'w') as f:
          try:
            json.dump(mym,f)
          except:
            pass
          f.close()
    #モーションの読み出し
    elif k == 'l':
      print('load motion file')
      if os.path.exists(motionfile):
        with open(motionfile,'r') as f:
          try:
            j = json.load(f)
            mym[:] = []
            for i in range(len(j)):
              mym.append(Pose(*j[i]))
          except:
            pass
          f.close()
      else:
        print (motionfile,' not found')

  # おしまい
  kb.set_normal_term()

  term = True;
  thread_1.join()

  for i in range(4):mt.stop(i)
  mt.enablectrl(IDs8, False)
  del mt

  DX2_ClosePort(dev)
  os._exit(0)
else:
  print('Could not open COM port.')
