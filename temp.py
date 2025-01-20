
import pickle

c = [1,11,1,1]
d = [2,2,2,2]
# print(a + b)
result = [a + b for a, b in zip(c, d)]
print(result)