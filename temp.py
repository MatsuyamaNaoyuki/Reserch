from myclass import myfunction

from myclass import Mydataset

testdatapath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortestnew.pickle"
traindatapath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit1500kaifortrain.pickle"

test_data,_,_ = myfunction.read_pickle_to_torch(testdatapath, True,True,True)

traindata,_,_ = myfunction.read_pickle_to_torch(traindatapath,True,True,True)

print(type(test_data))
mu_test, st_test = Mydataset.fit_standardizer_torch(test_data)
mu_train, st_train = Mydataset.fit_standardizer_torch(traindata)

myfunction.print_val(mu_test)
myfunction.print_val(mu_train)
myfunction.print_val(st_test)
myfunction.print_val(st_train)
