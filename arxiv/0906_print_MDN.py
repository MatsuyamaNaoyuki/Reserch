import matplotlib.pyplot as plt
from myclass import myfunction
import pandas as pd
import numpy as np
import torch





def culc_gosa(y_data, estimation_array):
    gosa = 0
    list_y_data = y_data[["Mc5x", "Mc5y", "Mc5z"]].values.tolist()
    npest = np.array([
        t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
        for t in estimation_array
    ])

    arr1 = npest[:, 0, -3:]   # 1つ目 → shape (6691, 12)
    arr2 = npest[:, 1, -3:]



    for i in range(len(y_data)):
        distance1 = np.linalg.norm(np.array(list_y_data[i])- np.array(arr1[i]))
        distance2 = np.linalg.norm(np.array(list_y_data[i])- np.array(arr2[i]))
        distance = min(distance1, distance2)
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
    

y_data = myfunction.load_pickle(r"D:\Matsuyama\laerningdataandresult\re3tubefingerforMDPI\mixhit10kaifortestbase.pickle")
estimation_array= myfunction.load_pickle(r"D:\Matsuyama\laerningdataandresult\re3tubefingerforMDPI\MDN\result20251006_123216.pickle")


no_contact_real = True
contact_real = True
no_contact_est = True
contact_est = True




list_estimation_array = [t.cpu().tolist() for t in estimation_array]

list1 = [row[0] for row in list_estimation_array]  # shape: (6871, 3)
list2 = [row[1] for row in list_estimation_array]

list_y_data = y_data[["Mc5x", "Mc5y", "Mc5z"]].values.tolist()







half = len(y_data) // 2

lx  = y_data['Mc5x'].to_list()
x = lx[half::1]
ly  = y_data['Mc5y'].to_list()
y = ly[half::1]
lz  = y_data['Mc5z'].to_list()
z = lz[half::1]

ex = [row[9]  for row in list1]
ey = [row[10] for row in list1]
ez = [row[11] for row in list1]

nx  = lx[:half:1]
ny  = ly[:half:1]
nz  = lz[:half:1]

enx = [row[9]  for row in list2]
eny = [row[10] for row in list2]
enz = [row[11] for row in list2]

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
