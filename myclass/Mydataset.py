import torch
from dataclasses import dataclass

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
    x_neutral = xdata[rise_mask]

    return x_neutral

# def fit_calibration_torch(x_neutral:torch.Tensor, x_ramp:torch.Tensor
#                           , ref_ramp:torch.Tensor|None=None, q_low:float=0.05, q_hi:float=0.95)->Calibfit:
    
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

def fit_standardizer_torch(x_train: torch.Tensor):
    mu = torch.nanmean(x_train, dim = 0)
    sd = torch.nanstd(x_train, dim = 0, unbiased = False)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    return mu, sd

def apply_standardized_torch(x: torch.Tensor, mu:torch.Tensor, sd:torch.Tensor):
    return (x-mu) / sd