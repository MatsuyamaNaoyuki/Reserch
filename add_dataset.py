import pandas as pd



file1_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger_mixhit\tube_softfinger_hit_1500_20250414_125109.pickle"
df1 = pd.read_pickle(file1_path)

file2_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger_mixhit\tube_softfinger_nohit_1500_20250411_144433.pickle"
df2 = pd.read_pickle(file2_path)



# データフレームを結合
merged_df = pd.concat([df1, df2], ignore_index=True)

# 結果を表示


output_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger_mixhit\tube_softfinger_mixhit_3000_20250411_144433.pickle"
merged_df.to_pickle(output_path)
