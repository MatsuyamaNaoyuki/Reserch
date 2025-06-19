import pandas as pd






file_path_list =[r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger0526\data\0526_tubefinger_hitfortest_time_10kai20250530_152508.pickle",
                 r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger0526\data\0526_tubefinger_nohitfortest_10kai20250530_152546.pickle"]

marge_df = None

for i in range (len(file_path_list)):
    df = pd.read_pickle(file_path_list[i])
    df['type'] = i
    if marge_df is None:
        marge_df = df
    else:
        merged_df = pd.concat([marge_df, df], ignore_index=True)




output_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\tubefinger0526\data\mixhitfortest.pickle"
merged_df.to_pickle(output_path)
