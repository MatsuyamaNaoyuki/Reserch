import matplotlib.pyplot as plt
from myclass import myfunction
import pandas as pd
import numpy as np





def culc_gosa(y_data, estimation_array):
    gosa = 0

    for i in range(len(y_data)):
        distance = np.linalg.norm(np.array(y_data[i])- np.array(estimation_array[i]))
        gosa = gosa + distance


    gosa = gosa / len(y_data)

    myfunction.print_val(gosa)



    

y_data = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\mixhit10kaifortest.pickle")
estimation_array= myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\GRU_seikika_nosentor\result20250906_142354.pickle")
js = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\GRU_seikika_nosentor\js20250906_142355.pickle")


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

culc_gosa([[1,1,1],[1,1,1]], [[0,0,0],[0,0,0]])
