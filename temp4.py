import matplotlib.pyplot as plt
from myclass import myfunction
import pandas as pd

y_data = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortestnew.pickle")
estimation_array= myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\Reserch\result20250827_174221.pickle")



list_estimation_array = [t.cpu().tolist()for t in estimation_array]

half = len(list_estimation_array) // 2

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

myfunction.print_val(ex)
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
# axes[0].scatter(t, x, label="contact X", s=1, c="blue", alpha=0.7)
# axes[0].scatter(t, ex, label="Estimated contact X", s=1, c="red", alpha=0.7)
axes[0].scatter(t, nx, label="nocontact X", s=1, c="green", alpha=0.7)
axes[0].scatter(t, enx, label="Estimated nocontact X", s=1, c="orange", alpha=0.7)

axes[0].set_ylabel("X")
axes[0].legend()

# Y
# axes[1].scatter(t, y, label="contact y", s=1, c="blue", alpha=0.7)
# axes[1].scatter(t, ey, label="Estimated contact y", s=1, c="red", alpha=0.7)
axes[1].scatter(t, ny, label="nocontact y", s=1, c="green", alpha=0.7)
axes[1].scatter(t, eny, label="Estimated nocontact y", s=1, c="orange", alpha=0.7)
axes[1].set_ylabel("Y")
axes[1].legend()

# Z
# axes[2].scatter(t, z, label="contact z", s=1, c="blue", alpha=0.7)
# axes[2].scatter(t, ez, label="Estimated contact z", s=1, c="red", alpha=0.7)
axes[2].scatter(t, nz, label="nocontact z", s=1, c="green", alpha=0.7)
axes[2].scatter(t, enz, label="Estimated nocontact z", s=1, c="orange", alpha=0.7)
# axes[2].set_ylabel("Z")
axes[2].set_xlabel("Time step")
axes[2].legend()

plt.tight_layout()
plt.show()
