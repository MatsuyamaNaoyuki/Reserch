from myclass import myfunction
import pandas as pd


df = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\mixhit_fortraintypenan20250715_163007.pickle")

has_nan = df.isna().any().any()

print("NaNが含まれているか:", has_nan)

# NaN がある場所を確認したい場合（オプション）
print("NaNのある列ごとの数:")
print(df.isna().sum())