#---------------------------------------------
# Access by item name in control table
#---------------------------------------------
import collections, unimportant_program.dx2lib as dx2lib
from ctypes import *

# Minimum unit for each item
class _i8(Union):
  _pack_ = 1
  _fields_ = ('d',c_int8),('u',c_uint8 * 1)
class _u8(Union):
  _pack_ = 1
  _fields_ = ('d',c_uint8),('u',c_uint8 * 1)
class _i16(Union):
  _pack_ = 1
  _fields_ = ('d',c_int16),('u',c_uint8 * 2)
class _u16(Union):
  _pack_ = 1
  _fields_ = ('d',c_uint16),('u',c_uint8 * 2)
class _i32(Union):
  _pack_ = 1
  _fields_ = ('d',c_int32),('u',c_uint8 * 4)
class _u32(Union):
  _pack_ = 1
  _fields_ = ('d',c_uint32),('u',c_uint8 * 4)

# Supported Dynamixel
class _TdevPROTable(Structure):
  _pack_ = 1
  _fields_ = [
    ('ModelNumber', _u16 * 1),('ModelInformation', _u32 * 1),('FirmwareVersion', _u8 * 1),
    ('ID', _u8 * 1),('BaudRate', _u8 * 1),('ReturnDelayTime', _u8 * 1),
    ('reserve', _u8 * 1),
    ('OperatingMode', _u8 * 1),
    ('reserve', _u8 * 1),
    ('HomingOffset', _i32 * 1),('MovingThreshold', _u32 * 1),
    ('TemperatureLimit', _u8 * 1),('MaxVoltageLimit', _u16 * 1),('MinVoltageLimit', _u16 * 1),('AccelerationLimit', _u32 * 1),('TorqueLimit', _u16 * 1),('VelocityLimit', _u32 * 1),
    ('MaxPositionLimit', _i32 * 1),('MinPositionLimit', _i32 * 1),
    ('ExternalPortMode1', _u8 * 1),('ExternalPortMode2', _u8 * 1),('ExternalPortMode3', _u8 * 1),('ExternalPortMode4', _u8 * 1),
    ('Shutdown', _u8 * 1),
    ('IndirectAddress1_256', _u16 * 256),
    ('reserve', _u8 * 1),
    ('TorqueEnable', _u8 * 1),
    ('LEDRed', _u8 * 1),('LEDGreen', _u8 * 1),('LEDBlue', _u8 * 1),
    ('reserve', _u8 * 20),
    ('VelocityIGain', _u16 * 1),('VelocityPGain', _u16 * 1),
    ('reserve', _u8 * 4),
    ('PositionPGain', _u16 * 1),
    ('GoalPosition', _i32 * 1),('GoalVelocity', _i32 * 1),('GoalTorque', _i16 * 1),('GoalAcceleration', _i32 * 1),
    ('Moving', _u8 * 1),
    ('PresentPosition', _i32 * 1),('PresentVelocity', _i32 * 1),
    ('reserve', _u8 * 2),
    ('PresentCurrent', _i16 * 1),('PresentInputVoltage', _u16 * 1),('PresentTemperature', _u8 * 1),
    ('ExternalPortData1', _u16 * 1), ('ExternalPortData2', _u16 * 1),('ExternalPortData3', _u16 * 1),('ExternalPortData4', _u16 * 1),
    ('IndirectData1_256', _u8 * 256),
    ('RegisteredInstruction', _u8 * 1),('StatusReturnLevel', _u8 * 1),('HardwareErrorStatus', _u8 * 1),
  ]
  def __str__(self):
    return "{}: {{{}}}".format(self.__class__.__name__, ", ".join(["{}: {}".format(field[0],getattr(self,field[0])) for field in self._fields_]))

class _TdevPROPTable(Structure):
  _pack_ = 1
  _fields_ = [
    ('ModelNumber', _u16 * 1),('ModelInformation', _u32 * 1),('FirmwareVersion', _u8 * 1),
    ('ID', _u8 * 1),('BaudRate', _u8 * 1),('ReturnDelayTime', _u8 * 1),
    ('DriveMode', _u8 * 1),('OperatingMode', _u8 * 1),
    ('SecondaryID', _u8 * 1),
    ('ProtocolType', _u8 * 1),
    ('reserve', _u8 * 6),
    ('HomingOffset', _i32 * 1),('MovingThreshold', _u32 * 1),
    ('reserve', _u8 * 3),
    ('TemperatureLimit', _u8 * 1),('MaxVoltageLimit', _u16 * 1),('MinVoltageLimit', _u16 * 1),('PWMLimit', _u16 * 1),('CurrentLimit', _u16 * 1),('AccelerationLimit', _u32 * 1),('VelocityLimit', _u32 * 1),
    ('MaxPositionLimit', _i32 * 1),('MinPositionLimit', _i32 * 1),
    ('ExternalPortMode1', _u8 * 1),('ExternalPortMode2', _u8 * 1),('ExternalPortMode3', _u8 * 1),('ExternalPortMode4', _u8 * 1),
    ('StartupConfiguration', _u8 * 1),
    ('reserve', _u8 * 2),
    ('Shutdown', _u8 * 1),
    ('reserve', _u8 * 104),
    ('IndirectAddress1_128', _u16 * 128),
    ('reserve', _u8 * 88),
    ('TorqueEnable', _u8 * 1),('LEDRed', _u8 * 1),('LEDGreen', _u8 * 1),('LEDBlue', _u8 * 1),
    ('StatusReturnLevel', _u8 * 1),('RegisteredInstruction', _u8 * 1),('HardwareErrorStatus', _u8 * 1),
    ('reserve', _u8 * 5),
    ('VelocityIGain', _u16 * 1),('VelocityPGain', _u16 * 1),('PositionDGain', _u16 * 1),('PositionIGain', _u16 * 1),('PositionPGain', _u16 * 1),
    ('reserve', _u8 * 2),
    ('Feedforward2ndGain', _u16 * 1),('Feedforward1stGain', _u16 * 1),
    ('reserve', _u8 * 6),
    ('BusWatchdog', _u8 * 1),
    ('reserve', _u8 * 1),
    ('GoalPWM', _i16 * 1),('GoalCurrent', _i16 * 1),('GoalVelocity', _i32 * 1),('ProfileAcceleration', _u32 * 1),('ProfileVelocity', _u32 * 1), ('GoalPosition', _i32 * 1),
    ('RealtimeTick', _u16 * 1),
    ('Moving', _u8 * 1),('MovingStatus', _u8 * 1),
    ('PresentPWM', _i16 * 1),('PresentCurrent', _i16 * 1),('PresentVelocity', _i32 * 1),('PresentPosition', _i32 * 1),
    ('VelocityTrajectory', _u32 * 1),('PositionTrajectory', _u32 * 1),
    ('PresentInputVoltage', _u16 * 1),('PresentTemperature', _u8 * 1),
    ('reserve', _u8 * 5),
    ('ExternalPortData1', _u16 * 1),('ExternalPortData2', _u16 * 1),('ExternalPortData3', _u16 * 1),('ExternalPortData4', _u16 * 1),
    ('reserve', _u8 * 26),
    ('IndirectData1_128', _u8 * 128),
    ('reserve', _u8 * 116),
    ('BackupReady', _u8 * 1),
  ]
  def __str__(self):
    return "{}: {{{}}}".format(self.__class__.__name__, ", ".join(["{}: {}".format(field[0],getattr(self,field[0])) for field in self._fields_]))

class _TdevXTable(Structure):
  _pack_ = 1
  _fields_ = [
    ('ModelNumber', _u16 * 1),('ModelInformation', _u32 * 1),('FirmwareVersion', _u8 * 1),
    ('ID', _u8 * 1),('Baudrate', _u8 * 1),('ReturnDelayTime', _u8 * 1),
    ('DriveMode', _u8 * 1),('OperatingMode', _u8 * 1),
    ('SecondaryID', _u8 * 1),
    ('ProtocolType', _u8 * 1),
    ('reserve', _u8 * 6),
    ('HomingOffset', _i32 * 1),('MovingThreshold', _u32 * 1),
    ('reserve', _u8 * 3),
    ('TemperatureLimit', _u8 * 1),('MaxVoltageLimit', _u16 * 1),('MinVoltageLimit', _u16 * 1),('PWMLimit', _u16 * 1),('CurrentLimit', _u16 * 1),('AccelerationLimit', _u32 * 1),('VelocityLimit', _u32 * 1),
    ('MaxPositionLimit', _u32 * 1),('MinPositionLimit',_u32 * 1),
    ('ExternalPortMode1', _u8 * 1),('ExternalPortMode2', _u8 * 1),('ExternalPortMode3', _u8 * 1),
    ('reserve', _u8 * 1),
    ('StartupConfiguration', _u8 * 1),
    ('reserve', _u8 * 2),
    ('Shutdown', _u8 * 1),('TorqueEnable', _u8 * 1),('LED', _u8 * 1),
    ('reserve', _u8 * 2),
    ('StatusReturnLevel', _u8 * 1),('RegisteredInstruction', _u8 * 1),('HardwareErrorStatus', _u8 * 1),
    ('reserve', _u8 * 5),
    ('VelocityIGain', _u16 * 1),('VelocityPGain', _u16 * 1),('PositionDGain', _u16 * 1),('PositionIGain', _u16 * 1),('PositionPGain', _u16 * 1),
    ('reserve', _u8 * 2),
    ('Feedforward2ndGain', _u16 * 1),('Feedforward1stGain', _u16 * 1),
    ('reserve', _u8 * 6),
    ('BusWatchdog', _i8 * 1),
    ('reserve', _u8 * 1),
    ('GoalPWM', _i16 * 1),('GoalCurrent', _i16 * 1),('GoalVelocity', _i32 * 1),
    ('ProfileAcceleration', _u32 * 1),('ProfileVelocity', _u32 * 1),
    ('GoalPosition', _i32 * 1),
    ('RealtimeTick', _u16 * 1),
    ('Moving', _u8 * 1),('MovingStatus', _u8 * 1),
    ('PresentPWM', _i16 * 1),('PresentCurrent', _i16 * 1),('PresentVelocity', _i32 * 1),('PresentPosition', _i32 * 1),
    ('VelocityTrajectory', _u32 * 1),('PositionTrajectory', _u32 * 1),
    ('PresentInputVoltage', _u16 * 1),('PresentTemperature', _u8 * 1),('BackupReady', _u8 * 1),
    ('reserve', _u8 * 4),
    ('ExternalPortData1', _u16 * 1),
    ('ExternalPortData2', _u16 * 1),
    ('ExternalPortData3', _u16 * 1),
    ('reserve', _u8 * 10),
    ('IndirectAddress1_28', _u16 * 28),
    ('IndirectData1_28', _u8 * 28),
    ('reserve', _u8 * 326),
    ('IndirectAddress29_56', _u16 * 28),
    ('IndirectData29_56', _u8 * 28),
  ]
  def __str__(self):
    return "{}: {{{}}}".format(self.__class__.__name__, ", ".join(["{}: {}".format(field[0],getattr(self,field[0])) for field in self._fields_]))

class _TdevX330Table(Structure):
  _pack_ = 1
  _fields_ = [
    ('ModelNumber', _u16 * 1),('ModelInformation', _u32 * 1),('FirmwareVersion', _u8 * 1),
    ('ID', _u8 * 1),('Baudrate', _u8 * 1),('ReturnDelayTime', _u8 * 1),
    ('DriveMode', _u8 * 1),('OperatingMode', _u8 * 1),
    ('SecondaryID', _u8 * 1),
    ('ProtocolType', _u8 * 1),
    ('reserve', _u8 * 6),
    ('HomingOffset', _i32 * 1),('MovingThreshold', _u32 * 1),
    ('reserve', _u8 * 3),
    ('TemperatureLimit', _u8 * 1),('MaxVoltageLimit', _u16 * 1),('MinVoltageLimit', _u16 * 1),('PWMLimit', _u16 * 1),('CurrentLimit', _u16 * 1),('AccelerationLimit', _u32 * 1),('VelocityLimit', _u32 * 1),
    ('MaxPositionLimit', _u32 * 1),('MinPositionLimit',_u32 * 1),
    ('ExternalPortMode1', _u8 * 1),('ExternalPortMode2', _u8 * 1),('ExternalPortMode3', _u8 * 1),
    ('reserve', _u8 * 1),
    ('StartupConfiguration', _u8 * 1),
    ('reserve', _u8 * 2),
    ('Shutdown', _u8 * 1),('TorqueEnable', _u8 * 1),('LED', _u8 * 1),
    ('reserve', _u8 * 2),
    ('StatusReturnLevel', _u8 * 1),('RegisteredInstruction', _u8 * 1),('HardwareErrorStatus', _u8 * 1),
    ('reserve', _u8 * 5),
    ('VelocityIGain', _u16 * 1),('VelocityPGain', _u16 * 1),('PositionDGain', _u16 * 1),('PositionIGain', _u16 * 1),('PositionPGain', _u16 * 1),
    ('reserve', _u8 * 2),
    ('Feedforward2ndGain', _u16 * 1),('Feedforward1stGain', _u16 * 1),
    ('reserve', _u8 * 6),
    ('BusWatchdog', _i8 * 1),
    ('reserve', _u8 * 1),
    ('GoalPWM', _i16 * 1),('GoalCurrent', _i16 * 1),('GoalVelocity', _i32 * 1),
    ('ProfileAcceleration', _u32 * 1),('ProfileVelocity', _u32 * 1),
    ('GoalPosition', _i32 * 1),
    ('RealtimeTick', _u16 * 1),
    ('Moving', _u8 * 1),('MovingStatus', _u8 * 1),
    ('PresentPWM', _i16 * 1),('PresentCurrent', _i16 * 1),('PresentVelocity', _i32 * 1),('PresentPosition', _i32 * 1),
    ('VelocityTrajectory', _u32 * 1),('PositionTrajectory', _u32 * 1),
    ('PresentInputVoltage', _u16 * 1),('PresentTemperature', _u8 * 1),('BackupReady', _u8 * 1),
    ('reserve', _u8 * 4),
    ('ExternalPortData1', _u16 * 1),
    ('ExternalPortData2', _u16 * 1),
    ('ExternalPortData3', _u16 * 1),
    ('reserve', _u8 * 10),
    ('IndirectAddress1_20', _u16 * 20),
    ('IndirectData1_20', _u8 * 20),
  ]
  def __str__(self):
    return "{}: {{{}}}".format(self.__class__.__name__, ", ".join(["{}: {}".format(field[0],getattr(self,field[0])) for field in self._fields_]))

class _TdevXL320Table(Structure):
  _pack_ = 1
  _fields_ = [
    ('ModelNumber', _u16 * 1),('FirmwareVersion', _u8 * 1),
    ('ID', _u8 * 1),('Baudrate', _u8 * 1),('ReturnDelayTime', _u8 * 1),
    ('CWAngleLimit', _u16 * 1),('CCWAngleLimit',_u16 * 1),
    ('reserve', _u8 * 1),
    ('ControlMode', _u8 * 1),
    ('TemperatureLimit', _u8 * 1),
    ('MinVoltageLimit', _u8 * 1),('MaxVoltageLimit', _u8 * 1),('MaxTorque', _u16 * 1),
    ('StatusReturnLevel', _u8 * 1),
    ('Shutdown', _u8 * 1),
    ('reserve', _u8 * 5),
    ('TorqueEnable', _u8 * 1),('LED', _u8 * 1),
    ('reserve', _u8 * 1),
    ('DGain', _u8 * 1),('IGain', _u8 * 1),('PGain', _u8 * 1),
    ('GoalPosition', _i16 * 1),('MovingSpeed', _u16 * 1),
    ('reserve', _u8 * 1),
    ('TorqueLimit', _u16 * 1),
    ('PresentPosition', _i16 * 1),('PresentSpeed', _i16 * 1),('PresentLoad', _i16 * 1),
    ('reserve', _u8 * 2),
    ('PresentVoltage', _i8 * 1),('PresentTemperature', _u8 * 1),
    ('RegisteredInstruction', _u8 * 1),
    ('reserve', _u8 * 1),
    ('Moving', _u8 * 1),('HardwareErrorStatus', _u8 * 1),
    ('Punch', _u16 * 1)
  ]
  def __str__(self):
    return "{}: {{{}}}".format(self.__class__.__name__, ", ".join(["{}: {}".format(field[0],getattr(self,field[0])) for field in self._fields_]))


class dxl:
  def __del__(self):
    pass

  def __init__(self, dev, id):
    """
    Init dxl object.
    

    Parameters
    ----------
    - dev (c_void_p): Handle on dx2lib initialization
    - id (int): Dynamixel ID to be linked
    """
    self._dev = dev
    self._id = id
    self._ctbl = None
    self._modelname = ''

    _dat = c_uint16(0)
    if dx2lib.DX2_ReadWordData(self._dev, self._id, 0, _dat, None):
      p = dx2lib.DXL_GetModelInfoByModelNo(int(_dat.value))
      if p:
        self._modelname = str(p.contents.name, 'utf-8')
        match p.contents.devtype:
          case 7:
            self._ctbl = _TdevPROTable()
          case 8:
            self._ctbl = _TdevPROPTable()
          case 9:
            if ('XC330' in self._modelname) or ('XL330' in self._modelname):
              self._ctbl = _TdevX330Table()
            else:
              self._ctbl = _TdevXTable()
        if self._ctbl:
          self.update()
        else:
          self._modelname = ''
          self._ctrl = None

  @property
  def __str__(self):
    return self._ctbl.__str__();

  @property
  def id(self):
    return self._id

  @property
  def modelname(self):
    return self._modelname

  def __iteminfo(self, name):
    m = self._ctbl.__class__.__dict__[name]
    if m:
      num = 0
      t_sz = 0
      typ = type(m.__get__(self._ctbl))._type_()
      match typ:
        case _i8():
          t_sz = 1
          num = m.size
        case _u8():
          t_sz = 1
          num = m.size
        case _i16():
          t_sz = 2
          num = m.size >> 1
        case _u16():
          t_sz = 2
          num = m.size >> 1
        case _i32():
          t_sz = 4
          num = m.size >> 2
        case _u32():
          t_sz = 4
          num = m.size >> 2
        case _:
          num = m.size
      return m, m.offset, typ, t_sz, num

  def itemaddress(self, name):
    """
    Get its address from the item name.

    Parameters
    ----------
    - name (str): item name
    """
    try:
      return self._ctbl.__class__.__dict__[name].offset
    except:
      pass
    return -1

  def itemname(self, addr):
    """
    Get its item name from the address.

    Parameters
    ----------
    - addr (int): item address
    """
    for m in self._ctbl._fields_:
      if (addr == self.itemaddress(m[0])) and (m[0] != 'reserve'):
        return m[0]

  def __read_bulk(self, param:()):
    l = 0
    for i, d in enumerate(param): l += d.length + 5
    rdat = (c_uint8 * l)()
    num = c_uint32(len(param))
    dx2lib.DX2_ReadBulkData(self._dev, param, num, rdat, None)
    i = 0
    r = ()
    while i < l:
      s = rdat[i + 0] | (rdat[i + 1] >> 1)
      if s == 0: break
      id = rdat[i + 2]
      err = rdat[i + 3] | (rdat[i + 4] << 8)
      r += (id, (rdat[i + 5: i + s])),
      i += s
    return r

  def get_bulk(self, dats:()):
    """
    Get data from multiple devices using ReadBulk instruction.
    Returns value from the item name.

    Parameters
    ----------
    - dats (list|tuple): (ID, "ItemName"), ...
    
    """
    bulkdat = (dx2lib.TBulkReadParam * len(dats))()
    for i, dat in enumerate(dats):
      bulkdat[i].id = dat[0]
      n, bulkdat[i].addr, typ, t_sz, num = self.__iteminfo(dat[1])
      bulkdat[i].length = t_sz * num
    r = self.__read_bulk(bulkdat)
    result = ()
    for dat in dats:
      for _r in r:
        if _r[0] == dat[0]:
          n, ofs, typ, t_sz, num = self.__iteminfo(dat[1])
          uu = (c_uint8 * (t_sz * num))(*_r[1])
          match typ:
            case _i8(): d = (_i8 * num)()
            case _u8(): d = (_u8 * num)()
            case _i16(): d = (_i16 * num)()
            case _u16(): d = (_u16 * num)()
            case _i32(): d = (_i32 * num)()
            case _u32(): d = (_u32 * num)()
          memmove(byref(d, 0), byref(uu, 0), t_sz * num)

          if num > 2:
            ret = ()
            for v in d:
              ret += v.d,
          else:
            ret = d[0].d
          result += (ret),
          break
      else:
        result += (),
    return result

  def set_bulk(self, dats:()):
    """
    Set data to multiple devices using WriteBulk instruction.
    Returns API's result.
    However, the actual success or failure of writing to the device is not determined.

    Parameters
    ----------
    - dats (list|tuple): (ID, "ItemName", Value), ...
    
    """
    wbulkdat = []
    if dats:
      ids = ()
      for dat in dats:
        ids += dat[0],
        if self.itemaddress(dat[1]) >= 0:
          n, ofs, typ, t_sz, num = self.__iteminfo(dat[1])
          if n:
            td = ()
            if isinstance(dat[2], (tuple, list)): td = dat[2]
            else: td = dat[2],
            sz = t_sz * num
            bsz = 0
            wdat = (c_uint8 * (sz + 5))(dat[0], ofs & 0xff, (ofs >> 8) & 0xff)
            match typ:
              case _i8():
                dat = (_i8 * num)()
                for i, v in enumerate(zip(dat, td)):
                  dat[i].d = c_int8(v[1])
                  for j in range(t_sz):
                    wdat[i + 5 + j] = dat[i].u[j]
                    bsz += 1
              case _u8():
                dat = (_u8 * num)()
                for i, v in enumerate(zip(dat, td)):
                  dat[i].d = c_uint8(v[1])
                  for j in range(t_sz):
                    wdat[i + 5 + j] = dat[i].u[j]
                    bsz += 1
              case _i16():
                dat = (_i16 * num)()
                for i, v in enumerate(zip(dat, td)):
                  dat[i].d = c_int16(v[1])
                  for j in range(t_sz):
                    wdat[i * 2 + 5 + j] = dat[i].u[j]
                    bsz += 1
              case _u16():
                dat = (_u16 * num)()
                for i, v in enumerate(zip(dat, td)):
                  dat[i].d = c_uint16(v[1])
                  for j in range(t_sz):
                    wdat[i * 2 + 5 + j] = dat[i].u[j]
                    bsz += 1
              case _i32():
                dat = (_i32 * num)()
                for i, v in enumerate(zip(dat, td)):
                  dat[i].d = c_int32(v[1])
                  for j in range(t_sz):
                    wdat[i * 4 + 5 + j] = dat[i].u[j]
                    bsz += 1
              case _u32():
                dat = (_u32 * num)()
                for i, v in enumerate(zip(dat, td)):
                  dat[i].d = c_uint32(v[1])
                  for j in range(t_sz):
                    wdat[i * 4 + 5 + j] = dat[i].u[j]
                    bsz += 1
              case _:
                return None
            wdat[3] = bsz & 0xff
            wdat[4] = (bsz >> 8) & 0xff
            wbulkdat += wdat[0:bsz + 5]

      if len(ids) != len(set(ids)): return None

      if len(wbulkdat) > 0:
        wbary = (c_uint8 * len(wbulkdat))(*wbulkdat)
        return dx2lib.DX2_WriteBulkData(self._dev, wbary, len(wbary), None)

  def update(self):
    """
    Receive information on all items on the control table.

    Parameters
    ----------
    """
    sz = sizeof(self._ctbl)
    t = (c_uint8 * sz)(0)
    if dx2lib.DX2_ReadBlockData(self._dev, self._id, 0, t, sz, None):
      memmove(POINTER(c_uint8)(self._ctbl), t, sz)

  def _get(self, name, upd):
    if self.itemaddress(name) >= 0:
      n, ofs, typ, t_sz, num = self.__iteminfo(name)
      if n:
        sz = t_sz * num
        if upd:
          rb = (c_uint8 * sz)(0)
          if dx2lib.DX2_ReadBlockData(self._dev, self._id, ofs, rb, sz, None):
            memmove(byref(self._ctbl, ofs), rb, sz)
        if num == 1:
          return (n.__get__(self._ctbl))[0].d
        else:
          r = ()
          for i in range(num): r += n.__get__(self._ctbl)[i].d,
          return r

  def get(self, name, update = False):
    """
    Get its value from the item name.
    Returns value from the item name.

    Parameters
    ----------
    - name (str): item name
    - update (bool): If true, it communicates with dynamiel to get data
    """
    if name:
      if isinstance(name, (tuple, list)):
        result = ()
        for m in name: result += self._get(m, update),
        return result
      else:
        return self._get(name, update)

  def _set(self, name, d):
    if self.itemaddress(name) >= 0:
      n, ofs, typ, t_sz, num = self.__iteminfo(name)
      if n:
        td = ()
        if isinstance(d, (tuple, list)): td = d
        else: td = d,
        sz = t_sz * num
        match typ:
          case _i8():
            dat = (_i8 * num)()
            for i, v in enumerate(zip(dat, td)): dat[i].d = c_int8(v[1])
          case _u8():
            dat = (_u8 * num)()
            for i, v in enumerate(zip(dat, td)): dat[i].d = c_uint8(v[1])
          case _i16():
            dat = (_i16 * num)()
            for i, v in enumerate(zip(dat, td)): dat[i].d = c_int16(v[1])
          case _u16():
            dat = (_u16 * num)()
            for i, v in enumerate(zip(dat, td)): dat[i].d = c_uint16(v[1])
          case _i32():
            dat = (_i32 * num)()
            for i, v in enumerate(zip(dat, td)): dat[i].d = c_int32(v[1])
          case _u32():
            dat = (_u32 * num)()
            for i, v in enumerate(zip(dat, td)): dat[i].d = c_uint32(v[1])
          case _:
            return False

        if dx2lib.DX2_WriteBlockData(self._dev, self._id, ofs, cast(dat, POINTER(c_uint8)), sz, None):
          memmove(byref(self._ctbl, ofs), dat, sz)
          n.__set__(self._ctbl, dat)
          return True
      return False

  def set(self, *args):
    """
    Write data by specifying items and data.
    Returns decision.

    Parameters
    ----------
    - *args (list/tuple): (item name, data),
    """
    if len(args) == 2:
      return self._set(args[0], args[1])
    elif isinstance(args[0], (tuple, list)):
      result = ()
      for i in args[0]:
        m = i[0]
        d = i[1]
        result += (self._set(m, d),)
      return result
    return False
