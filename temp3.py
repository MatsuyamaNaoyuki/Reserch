from myclass import myfunction
import matplotlib.pyplot as plt
import pandas as pd

df = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_mixhit\mixhit_300020250225_204358.pickle")
df = pd.DataFrame(df)
# CSVとして保存（インデックス付き）
df.to_csv("output.csv", index=False, encoding='utf-8-sig')  # Excelでも文字化けしないようにutf-8-sig推奨


# print(df['time'].apply(type).value_counts())

# df['time'] = pd.to_datetime(df['time'])

# # グラフを描画
# plt.figure(figsize=(10, 4))
# plt.plot(df.index, df['time'], label='Time')

# plt.xlabel('Row Index')
# plt.ylabel('Timestamp')
# plt.title('Time Sequence Check')
# plt.grid(True)
# plt.tight_layout()
# plt.show()