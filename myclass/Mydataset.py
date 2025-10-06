import torch
from dataclasses import dataclass
from myclass import myfunction
@dataclass
class Calibfit:
    mu_neutral: torch.Tensor #静止時の平均
    sign: torch.Tensor #符号によって向きをそろえる
    p_low:torch.Tensor
    p_hi:torch.Tensor
    amp:torch.Tensor #有効振幅、これをもとに振れ幅を合わせる


def make_xramp_and_reframp(xdata):
    motorleft = xdata[:, 0].float()
    dmotorleft = torch.diff(motorleft, prepend=motorleft[:1])

    rise_mask = (motorleft > 50) & (motorleft < 250) & (dmotorleft > 0.01)
    x_ramp = xdata[rise_mask]
    ref_ramp = motorleft[rise_mask]
    return x_ramp, ref_ramp

def make_xneutral(xdata):
    motor = xdata[:, 0:3].float()
    rise_mask = (motor[:, 0] < 1) & (motor[:, 1] < 1) & (motor[:, 2] < 1)
    rise_mask = trim_mask(rise_mask)
    x_neutral = xdata[rise_mask]
    return x_neutral

def trim_mask(rise_mask:torch.Tensor, max_gap:int = 10):
    result = torch.zeros_like(rise_mask, dtype=torch.bool)
    found_true = False
    gap = 0

    for i, val in enumerate(rise_mask):
        if val:
            result[i] = True
            found_true = True
            gap = 0

        elif found_true:
            gap = gap + 1
            if gap > max_gap:
                break

    return result


# def fit_calibration_torch(x_neutral:torch.Tensor, x_ramp:torch.Tensor
#                           , ref_ramp:torch.Tensor|None=None, q_low:float=0.05, q_hi:float=0.95)->Calibfit:
    

#ニュートラルの位置、レンジ等を求めてる
def fit_calibration_torch(calbedata: torch.Tensor, q_low:float=0.05, q_hi:float=0.95)->Calibfit:
    x_neutral = make_xneutral(calbedata)
    mu_neutral = torch.nanmean(x_neutral, dim=0)


    x_ramp, ref_ramp = make_xramp_and_reframp(calbedata)
    xr0 = x_ramp - mu_neutral
    ref = ref_ramp - torch.nanmean(ref_ramp)
    xr0_c = xr0 - torch.nanmean(xr0, dim=0, keepdim=True)
    cov = torch.nanmean(xr0_c * ref.unsqueeze(1), dim=0)
    sign = torch.where(cov >= 0, torch.ones_like(cov), -torch.ones_like(cov))


    calbedata0 = calbedata - mu_neutral
    xr = calbedata0 * sign

    p_low = torch.nanquantile(xr, q_low, dim=0)
    p_hi = torch.nanquantile(xr, q_hi, dim=0)
    amp = torch.clamp(p_hi - p_low, min=1e-8)

    return Calibfit(mu_neutral=mu_neutral, sign=sign, p_low=p_low, p_hi=p_hi, amp=amp)


def apply_align_torch(x:torch.Tensor, fit:Calibfit,
                       scale_alpha:torch.Tensor|float|None = None, add_mu_neutral:torch.Tensor | None = None) ->torch.Tensor:
    
    x0 = x - fit.mu_neutral
    x1 = x0 * fit.sign
    if scale_alpha is None:
        x2 = x1
    else:
        if isinstance(scale_alpha,float):
            scale = torch.full_like(x1[0], scale_alpha)
        else:
            scale = scale_alpha
        x2 = x1 * scale
    
    if add_mu_neutral is not None:
        x3 = x2 + add_mu_neutral
    else:
        x3 = x2

    return x3

def nanstd_t(x: torch.Tensor, dim=0, unbiased=False):
    m = torch.nanmean(x, dim=dim, keepdim=True)
    mask = ~torch.isnan(x)
    d = torch.where(mask, x - m, torch.zeros_like(x))
    sq = d * d
    cnt = mask.sum(dim=dim, keepdim=False)
    denom = (cnt - 1 if unbiased else cnt).clamp(min=1)
    var = sq.sum(dim=dim) / denom
    return torch.sqrt(var)

def fit_standardizer_torch(x_train: torch.Tensor):
    mu = torch.nanmean(x_train, dim = 0)
    sd = nanstd_t(x_train, dim = 0, unbiased = False)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    return mu, sd


def fit_normalizer_torch(x_train: torch.Tensor):
    mask = torch.isnan(x_train)
    x_masked = x_train.clone()
    x_masked[mask] = x_train.nanmean()
    max, _ = torch.max(x_masked, dim = 0)
    min,_ = torch.min(x_masked, dim = 0)
    scale = max - min
    return max, min, scale

def apply_standardize_torch(x: torch.Tensor, mu:torch.Tensor, sd:torch.Tensor):
    return (x-mu) / sd

def apply_normalize_torch(x: torch.Tensor, min:torch.Tensor, scale:torch.Tensor):
    return (x-min) / scale
def align_to_standardize_all(kijundata, data):

    fitdata = fit_calibration_torch(kijundata)

    alpha = torch.ones_like(fitdata.amp)
    data_prop = apply_align_torch(data, fitdata, alpha)

    mean, std = fit_standardizer_torch(data_prop)
    std_data = apply_standardize_torch(data_prop, mean, std)

    return std_data, mean, std, fitdata




def make_min_max(x_data, y_data, rotate = True, force = True, mag = True, category = True):
    x_inf = torch.where(torch.isnan(x_data), torch.tensor(float('inf'), device=x_data.device), x_data)
    x_minf = torch.where(torch.isnan(x_data), torch.tensor(float('-inf'), device=x_data.device), x_data)
    y_inf = torch.where(torch.isnan(y_data), torch.tensor(float('inf'), device=x_data.device), y_data)
    y_minf = torch.where(torch.isnan(y_data), torch.tensor(float('-inf'), device=x_data.device),y_data)    
    x_min = x_inf.amin(dim=0, keepdim=True)
    x_max = x_minf.amax(dim=0, keepdim=True)
    y_min = y_inf.amin(dim=0, keepdim=True)
    y_max = y_minf.amax(dim=0, keepdim=True)

    if category == False:
        return x_min, x_max, y_min, y_max  

    else: 
        rotatelen = 3
        forcelen = 3
        maglen = 9
        group_sizes = []
        if rotate:
            group_sizes.append(rotatelen)
        if force:
            group_sizes.append(forcelen)
        if mag:
            group_sizes.append(maglen)

        if sum(group_sizes) != len(x_data[0]):
            raise ValueError("長さが違うよ")


        result_parts_min = []
        result_parts_max = []
        start = 0

        for size in group_sizes:
            group_min = x_min[:, start:start+size].min()
            group_max = x_max[:, start:start+size].max()
            result_parts_min.append(group_min.repeat(size))
            result_parts_max.append(group_max.repeat(size))
            start = start + size
        x_min = torch.cat(result_parts_min, dim = 0)
        x_max = torch.cat(result_parts_max, dim = 0)



        return x_min, x_max, y_min, y_max