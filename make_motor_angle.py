import random
from myclass.MyDynamixel2 import MyDynamixel
import numpy as np
import datetime
from myclass import myfunction

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def culc_similarity(datas):
    motor_angle_base = np.array([150, 150, 150, 150])
    sim = 0
    for v1 in datas:
        sim = sim + cos_sim(motor_angle_base, v1)
    sim = sim / len(datas)
    return sim

    
    
# 変位幅をランダムで与える手法
def dlrandom():
    datas = []
    motor_max = 300
    motor_angle = np.array([150, 150, 150, 150])
    trial_num = 10
    i = 0
    while i < trial_num:
        change_i = random.randrange(4)
        change_range = random.uniform(-4, 4)
        if motor_angle[change_i] + change_range < motor_max and motor_angle[change_i] + change_range > 0:
            motor_angle[change_i] = motor_angle[change_i] + change_range

            #動かしたりの処理

            datas.append(motor_angle)  
            i = i + 1
    return datas

#ランダムに決めた値にランダムな幅で進んでいく手法
def lrandom():
    datas = []
    motor_max = 300
    trial_num = 1000
    i = 0
    start_angle = np.array([0,0,0,0])
    goal_angle = np.random.rand(4) * motor_max
    changestep =  [[] for _ in range(4)]
    for anglenum in range(4):
        changestep[anglenum] = complement_between(start_angle[anglenum], goal_angle[anglenum])
    changestep_list = [0,1,2,3] #changestepがおわってるか
    while(True):
        while changestep_list != []:
            for changestep_num in changestep_list:
                if changestep[changestep_num] == []:
                    changestep_list.remove(changestep_num)
                else:
                    if i > trial_num:
                        break 
                    start_angle[changestep_num] = changestep[changestep_num].pop(0)
                    datas.append(start_angle.copy())
                    i = i + 1
            else:
                continue
            break
        else:
            continue
        break
        
    return datas
 
        
#目的地を与えるとランダムに進んでいく幅を決める関数
def complement_between(a,b):
    anglelist = []
    max_change = 5
    angle = a
    if a == b:
        return [b]
    pm = (b - a) / abs(b-a)
    while pm * angle <  pm * b:
        change_range = random.uniform(1, 4)
        angle = angle + change_range * pm
        if angle * pm < pm * b:
            anglelist.append(angle)
    anglelist.append(b)
    return anglelist
    
def lrandom():
    datas = []
    motor_max = 300
    trial_num = 100
    i = 0
    start_angle = np.array([0,0,0,0])
    goal_angle = np.random.rand(4) * motor_max
    changestep =  [[] for _ in range(4)]
    for anglenum in range(4):
        changestep[anglenum] = complement_between(start_angle[anglenum], goal_angle[anglenum])
    changestep_list = [0,1,2,3] #changestepがおわってるか
    while(True):
        while changestep_list != []:
            for changestep_num in changestep_list:
                if changestep[changestep_num] == []:
                    changestep_list.remove(changestep_num)
                else:
                    if i > trial_num:
                        break 
                    start_angle[changestep_num] = changestep[changestep_num].pop(0)
                    datas.append(start_angle.copy())
                    i = i + 1
            else:
                continue
            break
        else:
            continue
        break
        
    return datas
 

def lfix_bet_randam():
    datas = []
    i = 0
    start_angle = np.array([0,0,0,0])
    angle = np.array([0,0,0,0])
    goal_anglelist = np.array([[300,0,0,0],[0,300,0,0],[0,0,300,0],[0,0,0,300],\
                              [300,300,0,0],[300,0,300,0],[300,0,0,300],[0,300,300,0],[0,300,0,300],[0,0,300,300],\
                              [0,300,300,300],[300,0,300,300],[300,300,0,300],[300,300,300,0],[300,300,300,300]])
    
    for num in range(len(goal_anglelist)):
        changestep =  [[] for _ in range(4)]
        for anglenum in range(4):
            changestep[anglenum] = complement_between(angle[anglenum], goal_anglelist[num][anglenum])
        changestep_list = [0,1,2,3] #changestepがおわってるか
        while changestep_list != []:
            for changestep_num in changestep_list:
                if changestep[changestep_num] == []:
                    changestep_list.remove(changestep_num)
                else:
                    angle[changestep_num] = changestep[changestep_num].pop(0)
                    datas.append(angle.copy())
                    i = i + 1

        changestep =  [[] for _ in range(4)]
        for anglenum in range(4):
            changestep[anglenum] = complement_between(angle[anglenum],start_angle[anglenum])
        changestep_list = [0,1,2,3] #changestepがおわってるか
        while changestep_list != []:
            for changestep_num in changestep_list:
                if changestep[changestep_num] == []:
                    changestep_list.remove(changestep_num)
                else:
                    angle[changestep_num] = changestep[changestep_num].pop(0)
                    datas.append(angle.copy())
                    i = i + 1
    
    print(i)
    return datas
 


def complement_between_with_maxchange(a, b, max_change):
    angle = a         # 現在の角度を a に設定
    angle_list = [a]
    try:
        pm = (b - a) / abs(b - a)  # 進行方向を決定 (+1 または -1)
    except ZeroDivisionError:
        return angle_list
    else:
        while pm * angle < pm * b:
            change = random.uniform(0, max_change)  # 変化量を 0～max_change の間でランダムに設定
            angle += change * pm                   # 現在の角度を更新
            angle_list.append(angle)


        if angle_list[-1] != b:
            angle_list.append(b)  # 最後に終了値 b を確実に追加
        return angle_list

def move_target_at_random(start_angles, goal):
    changesteps = []
    for i in range(len(goal)):
        changesteps.append(complement_between_with_maxchange(start_angles[i], goal[i], 5))
    max_length = max(len(row) for row in changesteps)
    padded_array = np.array([row + [row[-1]]* (max_length - len(row)) for row in changesteps])
    transposed_list = [[padded_array[row][col] for row in range(len(padded_array))]
                   for col in range(max_length)]
    return transposed_list

def move_and_return(goal):
    data = []
    data.extend(move_target_at_random([0,0,0,0], goal))
    data.extend(move_target_at_random(goal, [0,0,0,0]))
    return data
    

def randam_heimen():
    data = []
    for i in range(1):
        data.extend(move_target_at_random([0,0,0,0], [0,0,250,0]))
        data.extend(move_target_at_random([0,0,250,0], [0,0,0,0]))
    return data


def randam_rittai():
    data =[]
    left = 0
    right = 0
    while left < 240:
        left = left + random.uniform(15, 25)
        data.extend(move_and_return([left, 0, 250, 0]))
    while right < 240:
        right = right + random.uniform(15, 25)
        data.extend(move_and_return([0, right, 250, 0]))
    return data

        
def randam_rittai():
    data =[]
    left = 0
    right = 0
    while left < 240:
        left = left + random.uniform(15, 25)
        data.extend(move_and_return([left, 0, 250, 0]))
    while right < 240:
        right = right + random.uniform(15, 25)
        data.extend(move_and_return([0, right, 250, 0]))
    return data


def randam_rittai_fast():
    data =[]
    left = 0
    right = 0
    while left < 240:
        left = left + random.uniform(15, 25)
        data.extend([[left, 0, 250, 0]])
        data.extend([[0, 0, 0, 0]])
    while right < 240:
        right = right + random.uniform(15, 25)
        data.extend([[0, right, 250, 0]])
        data.extend([[0, 0, 0, 0]])
    return data


# motor_angle = randam_heimen()
# print(motor_angle)
motor_angle = []
for i in range(5):
    motor_angle.extend(randam_rittai_fast())
print(motor_angle)



filename = 'C:\\Users\\WRS\\Desktop\\Matsuyama\\laerningdataandresult\\0120_fast_3d\\houtomove_3d_fast'
myfunction.wirte_pkl(motor_angle, filename)
