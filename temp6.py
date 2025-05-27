from myclass import myfunction

df = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_GRU\mixhit_3000_with_type.pickle")
print(df[df['type'] == 0])
print(df[df['type'] == 1])