import torch
from dataclasses import dataclass

@dataclass
class Calibfit:
    mu_neutral: torch.Tensor #静止時の平均
    sign: torch.Tensor #符号によって向きをそろえる
    p_low:torch.Tensor
    p_hi:torch.Tensor
    amp:torch.Tensor #有効振幅、これをもとに振れ幅を合わせる

def fit_calibration_torch(x_neutral:torch.Tensor, x_ramp:torch.Tensor
                          , ref_ramp:torch.Tensor|None=None, q_low:float=0.05, q_hi:float=0.95)->Calibfit:
    
    mu_neutral = torch.nanmean(x_neutral, dim=0)

    xr0 = x_ramp - mu_neutral

    if ref_ramp is not None:
        ref = ref_ramp - torch.nanmean(ref_ramp)
        xr0_c = xr0 - torch.nanmean(xr0, dim=0, keepdim=True)
        cov = torch.nanmean(xr0_c * ref.unsqueeze(1), dim=0)
        sign = torch.where(cov >= 0, torch.ones_like(cov), -torch.ones_like(cov))
