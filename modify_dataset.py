import csv, pickle
from myclass import myfunction

# CSVファイルを開く



data = []
file_path = "1204_100data_maybeOK\\margedata20241205_194144.pickle"







try:
    with open(file_path, 'rb') as file:  # バイナリ読み込みモードで開く
        reader = pickle.load(file)         # データをロード
        data = list(reader)
except FileNotFoundError:
    print(f"ファイルが見つかりません: {file_path}")
except pickle.UnpicklingError:
    print("ピクルファイルのデコード中にエラーが発生しました。")




max_length = max(len(row) for row in data)

# print(max_length)
# 最大長に満たない行を削除
filtered_data = [row for row in data if len(row) == max_length]


filtered_data = [row for row in filtered_data if '' not in row]

filtered_data = [
    row for idx, row in enumerate(filtered_data)
    if (idx == 0) or all(500 <= int(row[i]) <= 800 for i in range(9, 18))  # 1行目は条件をスキップ
]

for i, row in enumerate(filtered_data):
    if i == 0:  # 1行目（インデックス0）をスキップ
        continue
    row[21] = row[21] - row[18]
    row[24] = row[24] - row[18]
    row[27] = row[27] - row[18]
    row[30] = row[30] - row[18]
    row[18] = row[18] - row[18]
    row[22] = row[22] - row[19]
    row[25] = row[25] - row[19]
    row[28] = row[28] - row[19]
    row[31] = row[31] - row[19]
    row[19] = row[19] - row[19]
    row[23] = row[23] - row[20]
    row[26] = row[26] - row[20]
    row[29] = row[29] - row[20]
    row[32] = row[32] - row[20]
    row[20] = row[20] - row[20]

for row in data:
    del row[18:21]

sukunaidata = []


for i, row in enumerate(filtered_data):
    if i % 100 == 0:
        sukunaidata.append(row)



myfunction.wirte_csv(sukunaidata, "1204_100data_maybeOK\\test")