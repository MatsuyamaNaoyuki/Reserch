from myclass import myfunction
import math

y_data = myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortestnew.pickle")
estimation_array= myfunction.load_pickle(r"C:\Users\WRS\Desktop\Matsuyama\Reserch\result20250827_174221.pickle")

list_estimation_array = [t.cpu().tolist()for t in estimation_array]
half = len(list_estimation_array) // 2

lx  = y_data['Mc5x'].to_list()
ly  = y_data['Mc5y'].to_list()
lz  = y_data['Mc5z'].to_list()

ex = [row[0]  for row in list_estimation_array]
ey = [row[1] for row in list_estimation_array]
ez = [row[2] for row in list_estimation_array]



distance = 0
for i in range(len(lx)):
    distance = math.sqrt((lx[i] - ex[i]) ** 2 + (ly[i] - ey[i]) ** 2 + (lz[i] - ez[i]) ** 2 )+ distance

distance = distance / len(lx)
myfunction.print_val(distance)