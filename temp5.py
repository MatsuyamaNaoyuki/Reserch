import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from myclass import myfunction

filepath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\with_hit_no_hit20250224_142757.pickle"
df = myfunction.load_pickle(filepath)

df['type'] = 1
print(df)






output_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\no_hit_for_train.pickle"
df.to_pickle(output_path)
