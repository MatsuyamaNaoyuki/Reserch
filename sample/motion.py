#---------------------------------------------
# Class to play motion
#---------------------------------------------
import ctypes, time, threading, collections, dx2lib

# Tuple for one pattern of motion
Pose = collections.namedtuple ('Pose', ('targetvalues', 'time'))

class motion:
  def __init__(self, dev, maxth = 4):
    self.__next_t = [time.time()] * maxth
    self.__motion = [None] * maxth
    self.__index = [0] * maxth
    self.__thread = [None] * maxth
    self.__dev = dev
    self.__ids = [()] * maxth
    self.__maxth = maxth

  def __del__(self):
    for i in range(self.__maxth):
      self.stop(i)

  def __run(self, *arg):
    no = arg[0]
    if self.__maxth > no:
      if len(self.__motion[no]) >= self.__index[no] + 1:
        m = self.__motion[no][self.__index[no]]
        arr = (ctypes.c_double * len(self.__ids[no]))(*m.targetvalues[0:len(self.__ids[no])])
        dx2lib.DXL_SetGoalAnglesAndTime2(self.__dev, self.__ids[no], arr, len(self.__ids[no]), m.time)
        self.__next_t[no] += m.time
        self.__index[no] += 1
        if (self.__index[no] <= len(self.__motion[no])):
          self.__thread[no] = threading.Timer(self.__next_t[no] - time.time(), self.__run, [no])
          self.__thread[no].start()
      else:
        self.__stop(no);

  def __stop(self, no):
    if self.__maxth > no:
      if self.__thread[no] != None:
        self.__thread[no].cancel()
        self.__thread[no] = None
      return True

  def stop(self, no):
    if self.__stop(no):
      if len(self.__ids[no]) > 0:
        dx2lib.DXL_StandStillAngles (self.__dev, self.__ids[no], len(self.__ids[no]))

  def start(self, no, ids, motion):
    if self.__stop(no):
      if len(motion) > 0:
        self.__ids[no] = (ctypes.c_uint8 * len(ids))(*ids)
        self.__motion[no] = motion
        self.__index[no] = 0
        self.__next_t[no] = time.time()
        self.__run(no)

  def running(self, no):
    if self.__maxth > no:
      return self.__thread[no] != None
    else:
      return False

  def enablectrl(self, ids, en):
    c_ids = (ctypes.c_uint8 * len(ids))(*ids)
    dx2lib.DXL_SetTorqueEnablesEquival(self.__dev, c_ids, len(c_ids), en)
