import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel2 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
import queue
import pickle


def move(Motors, ):
    change_angle = [[0,0,250,0], [0,0,0,0], [250,0,0,0], [0,0,0,0],[0,250,0,0],[0,0,0,0]]
    len_angle = len(change_angle)
    print(change_angle)
    for i, angles in enumerate(change_angle):
        print(angles)
        print(str(i) + "/" +  str(len_angle))
        Motors.move_to_points(angles, times = 7)



Motors = MyDynamixel()
move(Motors)
