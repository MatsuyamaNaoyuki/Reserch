from myclass import myfunction
import pandas as pd


path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\re3tubefinger0912\mixhit10kaifortest.pickle"


data = myfunction.load_pickle(path)
myfunction.print_val(data["sensor1"].dtype)


mask = data["type"].ne(data["type"].shift(fill_value=data["type"].iloc[0]))

# True になった行のインデックスを取り出す
first_indices = data.index[mask].tolist()

df1 = data.iloc[:first_indices[0]]
df1 = df1.reset_index(drop = True)
df2 = data.iloc[first_indices[0]:]
df2 = df2.reset_index(drop = True)

sensor_cols = df1.columns[df1.columns.str.contains("sensor")]

offsets = df1.loc[0, sensor_cols]
df1[sensor_cols] = (df1[sensor_cols] - offsets).astype(float)
offsets = df2.loc[0, sensor_cols]
df2[sensor_cols] = (df2[sensor_cols] - offsets).astype(float)
df_concat = pd.concat([df1, df2], axis=0, ignore_index=True)

myfunction.print_val(df_concat["sensor1"].dtype)

myfunction.wirte_pkl(df_concat, r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\re3tubefinger0912\mixhit10kaibase.pickle", time=False)