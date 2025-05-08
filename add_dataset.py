import pandas as pd






file_path_list =[r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_Diffusion\withhit_fortest20250227_134803.pickle",
                 r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_Diffusion\nohit_fortest20250227_134852.pickle"]

marge_df = None

for i in range (len(file_path_list)):
    df = pd.read_pickle(file_path_list[i])
    df['type'] = i
    if marge_df is None:
        marge_df = df
    else:
        merged_df = pd.concat([marge_df, df], ignore_index=True)




output_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_Diffusion\mixhit_fortest.pickle"
merged_df.to_pickle(output_path)
