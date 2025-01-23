import pandas as pd
import pickle

file_path = r"margedata20250123_124415.pickle"



df = pd.read_pickle(file_path)
df = pd.DataFrame(df)
print(len(df))
# df = df.drop(df.columns[0], axis=1)
# df = df[(df < 10000).all(axis=1)]
# print(len(df))