#位置制御を行うが，トルクを測定し，そこまでみたいなのを試す

import sys, time
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting


# ID一覧eq
IDs = (ctypes.c_uint8 * 4)(1,2,3,4)

#---------------------------------------------
dev = dx2.DX2_OpenPort(setting.COMPort, setting.Baudrate)
if dev != None:
  # ID一覧分のDynamixelを検索しモデルp名を表示
  for id in IDs:
    print(id, dx2.DXL_GetModelInfo(dev,id).contents.name.decode())

  # ID一覧分のDynamixelをMultiTurnモード=4に変更
  dx2.DXL_SetOperatingModesEquival(dev, IDs, len(IDs), 4)
  # ID一覧分のDynamixelをトルクディスエーブル
  dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), True)

  # キー入力により処理を分岐

  key = ''
  kb = kbhit.KBHit()
  nPWMs = (ctypes.c_double * len(IDs))()
  gPWMs = (ctypes.c_double * len(IDs))()
  nAngles = (ctypes.c_double * len(IDs))()
  gAngle = (ctypes.c_double)()
  
  dx2.DXL_GetPresentPWMs(dev, IDs, gPWMs, len(IDs))
  dx2.DXL_GetPresentAngle(dev, 4, gAngle)

  changeRange = 5
  changeAngle = 0

  while key != 'p':   # 'p'が押されると終了  
    if kb.kbhit():
      key = kb.getch()
      time.sleep(0.1)
  
  count = 0
  while count < 100:
    while(changeAngle < 301):
        gAngle.value = gAngle.value - changeRange
        changeAngle = changeAngle + changeRange
        dx2.DXL_SetGoalAngle (dev, 4,  gAngle)
        time.sleep(0.2)
    while(changeAngle > -1):
        gAngle.value = gAngle.value + changeRange
        changeAngle = changeAngle - changeRange
        dx2.DXL_SetGoalAngle (dev, 4,  gAngle)
        time.sleep(0.2)
    print(count)
    count = count + 1
    
    


  kb.set_normal_term()

  dx2.DXL_SetTorqueEnablesEquival(dev, IDs, len(IDs), False)

  dx2.DX2_ClosePort(dev)
else:
  print('Could not open COM port.')