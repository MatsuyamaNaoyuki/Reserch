from myclass import myfunction
import matplotlib.pyplot as plt
import pandas as pd

df = myfunction.load_pickle(r"C:\Users\shigf\Program\data\withhit\withhit\motor20250223_021713.pickle")
df = pd.DataFrame(df)

print(df)
