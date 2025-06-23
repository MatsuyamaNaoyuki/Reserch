from myclass import myfunction
import matplotlib.pyplot as plt
import pandas as pd

df = myfunction.load_pickle(r"C:\Users\shigf\Program\data\0520\hit1500kai\0526_tubefinger_hit_1500kai20250530_151330.pickle")
df = pd.DataFrame(df)

print(df)
