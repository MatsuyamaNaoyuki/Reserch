import pickle 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
from myclass import myfunction

# CSVファイルの読み込み
file_path = r"C:\Users\shigf\Program\Reserch\motor20250720_182716.pickle"
data =  myfunction.load_pickle(file_path)
print((len(data[0]) - 1) /2 )
print(data)


# x = range(len(data))

# # 2列目, 3列目, 4列目（1-based）→ 0-based index 1,2,3
# col2 = [row[1] for row in data]
# col3 = [row[2] for row in data]
# col4 = [row[3] for row in data]

# plt.plot(x, col2, label="col2")
# plt.plot(x, col3, label="col3")
# plt.plot(x, col4, label="col4")
# plt.xlabel("x")
# plt.ylabel("value")
# plt.legend()
# plt.tight_layout()
# plt.show()