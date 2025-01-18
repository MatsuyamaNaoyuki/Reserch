import pickle

i =2
with open("C:\\Users\\shigf\\Program\\data\\howtomove_0117_3d20250117_140223.pickle", mode='br') as fi:
    change_angle = pickle.load(fi)
print(change_angle[550])

print(str(i) + "/" +  str(len(change_angle)))