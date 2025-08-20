import pandas as pd






file_path_list =[r"C:\Users\shigf\Program\data\0818_tubefinger_rere\0818_tubefinger_hit_1500kai_rere20250819_080111.pickle",
                 r"C:\Users\shigf\Program\data\0818_tubefinger_rere\0818_tubefinger_nohit_1500kai_rere20250819_135940.pickle"]

marge_df = None

for i in range (len(file_path_list)):
    df = pd.read_pickle(file_path_list[i])
    df['type'] = i
    if marge_df is None:
        marge_df = df
    else:
        merged_df = pd.concat([marge_df, df], ignore_index=True)




output_path = r"C:\Users\shigf\Program\data\0818_tubefinger_rere\mixhit1500kaifortest.pickle"
merged_df.to_pickle(output_path)
