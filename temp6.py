from myclass import myfunction
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

df = myfunction.load_pickle(r"C:\Users\shigf\Program\data\0519_mech-newfinger\withhittest\howtomove_50020250127_164123.pickle")

print(len(df))
# df = pd.DataFrame(df)

# kensakutaisyou = df[['rotate1', 'rotate2', 'rotate3']]

# kensaku = kensakutaisyou[:700]


# arr1 = kensakutaisyou.to_numpy()  # shape: (100000, 3)
# arr2 = kensaku.to_numpy()  # shape: (700, 3)

# N = arr1.shape[0]
# L = arr2.shape[0]

# # 類似度格納用リスト
# similarities = []

# for i in tqdm(range(N - L + 1)):
#     window = arr1[i:i+L]  # (700, 3)
    
#     # ユークリッド距離（L2距離）の合計または平均
#     diff = window - arr2
#     distance = np.linalg.norm(diff, axis=1).mean()  # or sum()
    
#     similarities.append(distance)

# similarities = np.array(similarities)  # shape: (N - L + 1,)

# plt.figure(figsize=(12, 4))
# plt.plot(similarities, label='類似度（平均L2距離）')
# plt.xlabel('df1の開始インデックス')
# plt.ylabel('平均L2距離')
# plt.title('df2とdf1各部分との類似度')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()