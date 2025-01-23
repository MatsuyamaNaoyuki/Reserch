import pandas as pd



# file1_path = r"C:\Users\shigf\Program\data\sentan_newcam\modifydata20250122.csv"
# df1 = pd.read_csv(file1_path)

# file2_path = r"C:\Users\shigf\Program\data\sayuuhueruhazu\modifydata20250121.csv"
# df2 = pd.read_csv(file2_path)

# print(len(df2))


file2_path = r"C:\Users\shigf\Program\data\sentan_morecam\motioncapture20250122_190027.pickle"
df2 = pd.read_pickle(file2_path)

print(df2[0])
# [datetime.datetime(2025, 1, 20, 22, 7, 37, 729549)
# datetime.datetime(2025, 1, 20, 22, 19, 57, 432966)

# [datetime.datetime(2025, 1, 22, 17, 10, 55, 860427)