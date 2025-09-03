from myclass import myfunction
import torch

def nanstd_t(x: torch.Tensor, dim=0, unbiased=False):
    m = torch.nanmean(x, dim=dim, keepdim=True)
    mask = ~torch.isnan(x)
    d = torch.where(mask, x - m, torch.zeros_like(x))
    sq = d * d
    cnt = mask.sum(dim=dim, keepdim=False)
    denom = (cnt - 1 if unbiased else cnt).clamp(min=1)
    var = sq.sum(dim=dim) / denom
    return torch.sqrt(var)

big_marker_trainxdata,big_marker_trainydata,_ = myfunction.read_pickle_to_torch(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\mixhit1500kaifortrain.pickle",True,True,True)
small_marker_trainxdata,small_marker_trainydata,_ = myfunction.read_pickle_to_torch(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit1500kaifortrain.pickle",True,True,True)

big_xtrain_mean = big_marker_trainxdata.nanmean(dim=0, keepdim=True)
big_xtrain_std = nanstd_t(big_marker_trainxdata)
big_ytrain_mean = big_marker_trainydata.nanmean(dim=0, keepdim=True)
big_ytrain_std = big_marker_trainydata.std(dim=0, keepdim=True)

small_xtrain_mean = small_marker_trainxdata.nanmean(dim=0, keepdim=True)
small_xtrain_std = nanstd_t(small_marker_trainxdata)
small_ytrain_mean = small_marker_trainydata.nanmean(dim=0, keepdim=True)
small_ytrain_std = nanstd_t(small_marker_trainydata)

big_x_train_change = (big_marker_trainxdata - big_xtrain_mean) / big_xtrain_std
big_y_train_change = (big_marker_trainydata - big_ytrain_mean) / big_ytrain_std 
small_x_train_change = (small_marker_trainxdata - small_xtrain_mean) / small_xtrain_std
small_y_train_change = (small_marker_trainydata - small_ytrain_mean) / small_ytrain_std 


big_xtrainchange_mean = big_x_train_change.nanmean(dim=0, keepdim=True)
big_xtrainchange_std = nanstd_t(big_x_train_change)
big_ytrainchange_mean = big_y_train_change.nanmean(dim=0, keepdim=True)
big_ytrainchange_std = big_y_train_change.std(dim=0, keepdim=True)

small_xtrainchange_mean = small_x_train_change.nanmean(dim=0, keepdim=True)
small_xtrainchange_std = nanstd_t(small_x_train_change)
small_ytrainchange_mean = small_y_train_change.nanmean(dim=0, keepdim=True)
small_ytrainchange_std = nanstd_t(small_y_train_change)




big_marker_testxdata,big_marker_testydata,_ = myfunction.read_pickle_to_torch(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\retubefinger0816\mixhit10kaifortest.pickle",True,True,True)
small_marker_testxdata,small_marker_testydata,_ = myfunction.read_pickle_to_torch(r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortest (1).pickle",True,True,True)

big_xtest_mean = big_marker_testxdata.nanmean(dim=0, keepdim=True)
big_xtest_std = big_marker_testxdata.std(dim=0, keepdim=True)
big_ytest_mean = big_marker_testydata.nanmean(dim=0, keepdim=True)
big_ytest_std = big_marker_testydata.std(dim=0, keepdim=True)

small_xtest_mean = small_marker_testxdata.nanmean(dim=0, keepdim=True)
small_xtest_std = small_marker_testxdata.std(dim=0, keepdim=True)
small_ytest_mean = small_marker_testydata.nanmean(dim=0, keepdim=True)
small_ytest_std = small_marker_testydata.std(dim=0, keepdim=True)

big_x_test_change = (big_marker_testxdata - big_xtrain_mean) / big_xtrain_std
big_y_test_change = (big_marker_testydata - big_ytrain_mean) / big_ytrain_std 
small_x_test_change = (small_marker_testxdata - small_xtrain_mean) / small_xtrain_std
small_y_test_change = (small_marker_testydata - small_ytrain_mean) / small_ytrain_std 

big_xtestchange_mean = big_x_test_change.nanmean(dim=0, keepdim=True)
big_xtestchange_std = nanstd_t(big_x_test_change)
big_ytestchange_mean = big_y_test_change.nanmean(dim=0, keepdim=True)
big_ytestchange_std = big_y_test_change.std(dim=0, keepdim=True)

small_xtestchange_mean = small_x_test_change.nanmean(dim=0, keepdim=True)
small_xtestchange_std = nanstd_t(small_x_test_change)
small_ytestchange_mean = small_y_test_change.nanmean(dim=0, keepdim=True)
small_ytestchange_std = nanstd_t(small_y_test_change)

# myfunction.print_val(big_xtrainchange_mean)
# myfunction.print_val(big_xtrainchange_std)
# myfunction.print_val(big_ytrainchange_mean)
# myfunction.print_val(big_ytrainchange_std)
myfunction.print_val(big_xtestchange_mean)
myfunction.print_val(big_xtestchange_std)
myfunction.print_val(big_ytestchange_mean)
myfunction.print_val(big_ytestchange_std)
# myfunction.print_val(small_xtrainchange_mean)
# myfunction.print_val(small_xtrainchange_std)
# myfunction.print_val(small_ytrainchange_mean)
# myfunction.print_val(small_ytrainchange_std)
myfunction.print_val(small_xtestchange_mean)
myfunction.print_val(small_xtestchange_std)
myfunction.print_val(small_ytestchange_mean)
myfunction.print_val(small_ytestchange_std)

