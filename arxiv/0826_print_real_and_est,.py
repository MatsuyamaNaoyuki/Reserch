import matplotlib.pyplot as plt
from myclass import myfunction
import pandas as pd
import numpy as np





def culc_gosa(y_data, estimation_array):
    gosa = 0
    myfunction.print_val(len(y_data))
    myfunction.print_val(len(estimation_array))
    for i in range(len(y_data)):
        distance = np.linalg.norm(np.array(y_data[i])- np.array(estimation_array[i]))
        gosa = gosa + distance
    
    gosa = gosa / len(y_data)
    myfunction.print_val(gosa)

def culc_currect(y_data, estimation_array):
    half = len(y_data) // 2
    contact_y_data = y_data[:half]
    no_contact_y_data = y_data[half:]
    contact_y_data = contact_y_data + contact_y_data
    no_contact_y_data = no_contact_y_data + no_contact_y_data
    minlen = min(len(contact_y_data), len(no_contact_y_data), len(estimation_array))
    currect = 0
    for i in range(minlen):
        dis1 = np.linalg.norm(np.array(contact_y_data[i])- np.array(estimation_array[i]))
        dis2 = np.linalg.norm(np.array(no_contact_y_data[i])- np.array(estimation_array[i]))
        if i < half:
            if dis1 < dis2:
                currect = currect + 1
        else:
            if dis2 < dis1:
                currect = currect +1
    
    currect = currect / minlen
    myfunction.print_val(currect)
    

y_data = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortestnew.pickle")
estimation_array= myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\GRU_noforce\result20250902_144114.pickle")
estimation_array= myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\select2\result20250829_152315.pickle")

myfunction.print_val(estimation_array)
no_contact_real = True
contact_real = False
no_contact_est = True
contact_est = False


list_estimation_array = [t.cpu().tolist() for t in estimation_array]

# list1 = [row[0] for row in list_estimation_array]  # shape: (6871, 3)
# list2 = [row[1] for row in list_estimation_array]
list_y_data = y_data[["Mc5x", "Mc5y", "Mc5z"]].values.tolist()

culc_gosa(list_y_data, list_estimation_array)
culc_currect(list_y_data, list_estimation_array)





half = len(y_data) // 2

lx  = y_data['Mc5x'].to_list()
x = lx[half::1]
ly  = y_data['Mc5y'].to_list()
y = ly[half::1]
lz  = y_data['Mc5z'].to_list()
z = lz[half::1]

ex = [row[0]  for row in list_estimation_array[half::1]]
ey = [row[1] for row in list_estimation_array[half::1]]
ez = [row[2] for row in list_estimation_array[half::1]]

nx  = lx[:half:1]
ny  = ly[:half:1]
nz  = lz[:half:1]

enx = [row[0]  for row in list_estimation_array[:half:1]]
eny = [row[1] for row in list_estimation_array[:half:1]]
enz = [row[2] for row in list_estimation_array[:half:1]]

min_len = min(len(x), len(nx), len(ex), len(enx))
x, y, z = x[:min_len], y[:min_len], z[:min_len]
ex, ey, ez = ex[:min_len], ey[:min_len], ez[:min_len]
nx, ny, nz = nx[:min_len], ny[:min_len], nz[:min_len]
enx, eny, enz = enx[:min_len], eny[:min_len], enz[:min_len]


# 時系列インデックス
t = range(len(x))



print(len(z))
print(len(nx))

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# X

if contact_real:
    axes[0].scatter(t, nx, label="contact X", s=1, c="blue", alpha=0.7)  
    axes[1].scatter(t, ny, label="contact y", s=1, c="blue", alpha=0.7)
    axes[2].scatter(t, nz, label="contact z", s=1, c="blue", alpha=0.7)

if contact_est:
    axes[0].scatter(t, enx, label="Estimated contact X", s=1, c="red", alpha=0.7)
    axes[1].scatter(t, eny, label="Estimated contact y", s=1, c="red", alpha=0.7)
    axes[2].scatter(t, enz, label="Estimated contact z", s=1, c="red", alpha=0.7)


if no_contact_real:
    axes[0].scatter(t, x, label="nocontact X", s=1, c="green", alpha=0.7)
    axes[1].scatter(t, y, label="nocontact y", s=1, c="green", alpha=0.7)
    axes[2].scatter(t, z, label="nocontact z", s=1, c="green", alpha=0.7)

if no_contact_est:
    axes[0].scatter(t, ex, label="Estimated nocontact X", s=1, c="orange", alpha=0.7)
    axes[1].scatter(t, ey, label="Estimated nocontact y", s=1, c="orange", alpha=0.7)
    axes[2].scatter(t, ez, label="Estimated nocontact z", s=1, c="orange", alpha=0.7)
axes[0].set_ylabel("X")
axes[0].legend()

axes[1].set_ylabel("Y")
axes[1].legend()

axes[2].set_ylabel("Z")
axes[2].set_xlabel("Time step")
axes[2].legend()

plt.tight_layout()
plt.show()
