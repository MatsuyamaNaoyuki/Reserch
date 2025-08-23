from myclass import myfunction
import pandas as pd
import numpy as np

k = 11
max_k = 10

k = int(np.clip(k, 0, max_k))

myfunction.print_val(k)