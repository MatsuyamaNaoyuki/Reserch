import os
import pandas as pd

df1 = pd.read_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_Diffusion\mixhit_fortest.pickle")
df2 = pd.read_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_mixhit\mixhit_fortest20250227_135315.pickle")

print(df1)
print(df2)