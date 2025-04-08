import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import datetime

# CSVファイルの読み込み
file_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech\robomech300020250217_140437.pickle"
data = pd.read_pickle(file_path)

df = data[data.index  % 10 == 0]
filename = "robomech3000_10per"

now = datetime.datetime.now()
filename = filename + now.strftime('%Y%m%d_%H%M%S') + '.pickle'


df.to_pickle(filename)
