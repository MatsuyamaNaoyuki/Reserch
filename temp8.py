import matplotlib.pyplot as plt
from myclass import myfunction
import pandas as pd
import numpy as np
import pandas as pd
def culc_gosa(y_data, estimation_array):
    gosa = 0
    gosamae = 0
    gosaato = 0
    half = len(y_data)//2
    for i in range(len(y_data)):
        distance = np.linalg.norm(np.array(y_data[i])- np.array(estimation_array[i]))
        gosa = gosa + distance
        if i < half:
            gosamae = gosamae + distance
        else:
            gosaato = gosaato + distance

    gosa = gosa / len(y_data)
    gosaato = gosaato / half
    gosamae = gosamae / half
    myfunction.print_val(gosa)
    myfunction.print_val(gosaato)
    myfunction.print_val(gosamae)

# y_data = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\mixhit10kaifortest.pickle")
# estimation_array= myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\GRU_seikika_nosentor\result20250906_142354.pickle")
# js = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\GRU_seikika_nosentor\js20250906_142355.pickle")


y_data = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortestnew.pickle")
estimation_array= myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\GRUseikika_nosentor\result20250906_140348.pickle")
js = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\GRUseikika_nosentor\js20250906_140349.pickle")



idx = js.cpu().numpy()
y_data = y_data.iloc[idx]



no_contact_real = True
contact_real = False
no_contact_est = True
contact_est = False


list_estimation_array = [t.cpu().tolist() for t in estimation_array]



# list1 = [row[0] for row in list_estimation_array]  # shape: (6871, 3)
# list2 = [row[1] for row in list_estimation_array]
list_y_data = y_data[["Mc5x", "Mc5y", "Mc5z"]].values.tolist()

culc_gosa(list_y_data, list_estimation_array)




# DataFrameに変換してCSVに保存
df = pd.DataFrame(list_y_data, columns=["x", "y", "z"])
df.to_csv("y_small.csv", index=False, encoding="utf-8")
df = pd.DataFrame(list_estimation_array, columns=["x", "y", "z"])
df.to_csv("ey_small.csv", index=False, encoding="utf-8")