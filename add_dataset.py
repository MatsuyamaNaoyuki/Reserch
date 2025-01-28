import pandas as pd



file1_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\syougainasi\modifydata20250123.csv"
df1 = pd.read_csv(file1_path)

file2_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\sentan_morecam\modifydata20250122.csv"
df2 = pd.read_csv(file2_path)


half_len_df1 = len(df1) // 2
half_len_df2 = len(df2) // 2

# 前半分の行を取得
df1_half = df1.iloc[:half_len_df1]
df2_half = df2.iloc[:half_len_df2]

# データフレームを結合
merged_df = pd.concat([df1_half, df2_half], ignore_index=True)

# 結果を表示
df = merged_df[merged_df.index  % 10 == 0]

output_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\sentan&syougai\modifydata10per_20250123.pickle"
df.to_pickle(output_path)
