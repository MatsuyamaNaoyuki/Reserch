from myclass import myfunction
df = myfunction.load_pickle(r"C:\Users\shigf\Program\data\0520\hit\howtomove_5000_20250207_134537.pickle")

# df = myfunction.load_pickle(r"C:\Users\shigf\Downloads\mixhit_3000_with_type.pickle")
print(df)
df = df[10:3010]
print(df[-1])

# myfunction.wirte_pkl(df, "howtomove1500kai")
