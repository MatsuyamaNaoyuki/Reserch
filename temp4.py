import sys,os, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ctypes
from package import kbhit
from package import dx2lib as dx2
from package import setting
import csv
import pprint

dx2.DXL_SetGoalAnglesAndVelocities(self.dev, self.IDs, gole_angles, len(self.IDs))