import pandas as pd






file_path_list =[r"C:\Users\shigf\Program\data\withhit\withhit\withhit_fortrain20250225_202218.pickle",
                 r"C:\Users\shigf\Program\data\withhit\nohit\with_hit_no_hit20250224_142757.pickle"]

marge_df = None

for i in range (len(file_path_list)):
    df = pd.read_pickle(file_path_list[i])
    df['type'] = i
    if marge_df is None:
        marge_df = df
    else:
        merged_df = pd.concat([marge_df, df], ignore_index=True)




output_path = r"C:\Users\shigf\Program\data\testyou.pickle"
merged_df.to_pickle(output_path)
