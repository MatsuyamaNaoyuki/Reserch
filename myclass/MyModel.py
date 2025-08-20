import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import math

from typing import Union




class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self,input_dim, global_cond_dim, diffusion_step_embed_dim=256,down_dims=[256,512,1024], kernel_size=5, n_groups=8 ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually condition_horizon * condition_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,sample: torch.Tensor,timestep: Union[torch.Tensor, float, int],global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        # if not torch.is_tensor(timesteps):
        #     # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        #     timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        # elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        #     timesteps = timesteps[None].to(sample.device)
        if not isinstance(timesteps, torch.Tensor):  # ✅ TorchScript対応
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:  # ✅ JIT対応：len(timesteps.shape) は避ける
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):

            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)
    
class ResNet1D(nn.Module):
    def __init__(self, in_channels, base_width=64):
        super().__init__()
        self.inplanes = base_width
        self.conv1 = nn.Conv1d(in_channels, base_width, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1  = nn.BatchNorm1d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_width,   2, stride=1)
        self.layer2 = self._make_layer(base_width*2, 2, stride=2)
        self.layer3 = self._make_layer(base_width*4, 2, stride=2)
        self.layer4 = self._make_layer(base_width*8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)   # 出力 (B,512,1)
    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes))
        layers = [BasicBlock1D(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):                 # x: (B,C,L)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).squeeze(-1)   # (B,512)

class ResNetGRU(nn.Module):
    def __init__(self, input_dim, output_dim=12, hidden=128):
        super().__init__()
        self.backbone = ResNet1D(in_channels=input_dim)  # 1‑D ResNet (出力512)
        self.gru      = nn.GRU(512, hidden, batch_first=True)
        self.head     = nn.Linear(hidden, output_dim)


    def forward(self, x_seq):                 # (B,L,C)
        B, L, C = x_seq.shape

        # --- ★ 1. (B*L, C, 1) へ変形し「各時刻を独立に」ResNet に通す ---
        x_flat = x_seq.reshape(B*L, C).unsqueeze(-1)     # 長さ=1 の信号
        feat   = self.backbone(x_flat)                   # (B*L, 512)

        # --- ★ 2. 元の系列形に戻す
        feat   = feat.view(B, L, -1)                     # (B,L,512)

        # --- 3. GRU で時系列統合
        h, _   = self.gru(feat)                          # (B,L,hidden)

        

        return self.head(h[:, -1])    

class BasicBlock1Dforshap(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)

class ResNet1Dforshap(nn.Module):
    def __init__(self, in_channels, base_width=64):
        super().__init__()
        self.inplanes = base_width
        self.conv1 = nn.Conv1d(in_channels, base_width, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1  = nn.BatchNorm1d(base_width)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_width,   2, stride=1)
        self.layer2 = self._make_layer(base_width*2, 2, stride=2)
        self.layer3 = self._make_layer(base_width*4, 2, stride=2)
        self.layer4 = self._make_layer(base_width*8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)   # 出力 (B,512,1)
    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes))
        layers = [BasicBlock1Dforshap(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1Dforshap(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):                 # x: (B,C,L)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).squeeze(-1)   # (B,512)

class ResNetGRUforshap(nn.Module):
    def __init__(self, input_dim, output_dim=12, hidden=128):
        super().__init__()
        self.backbone = ResNet1Dforshap(in_channels=input_dim)  # 1‑D ResNet (出力512)
        self.gru      = nn.GRU(512, hidden, batch_first=True)
        self.head     = nn.Linear(hidden, output_dim)

    def forward(self, x_seq):                 # (B,L,C)
        B, L, C = x_seq.shape

        # --- ★ 1. (B*L, C, 1) へ変形し「各時刻を独立に」ResNet に通す ---
        x_flat = x_seq.reshape(B*L, C).unsqueeze(-1)     # 長さ=1 の信号
        feat   = self.backbone(x_flat)                   # (B*L, 512)

        # --- ★ 2. 元の系列形に戻す
        feat   = feat.view(B, L, -1)                     # (B,L,512)

        # --- 3. GRU で時系列統合
        h, _   = self.gru(feat)                          # (B,L,hidden)

        return self.head(h[:, -1])    
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ResNetTransformer(nn.Module):
    def __init__(self, input_dim, output_dim=12, d_model=512, nhead=8, nhid=2048, nlayers=2, dropout=0.1):
        super().__init__()
        self.backbone = ResNet1D(in_channels=input_dim)  # 出力は (B*L, 512)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x_seq):
        B, L, C = x_seq.shape  # (B, L, C)
        # ResNet に通す
        x_flat = x_seq.reshape(B*L, C).unsqueeze(-1)  # (B*L, C, 1)
        feat = self.backbone(x_flat)                 # (B*L, 512)
        feat = feat.view(B, L, -1)                  # (B, L, 512)
        # Positional Encoding
        feat = self.pos_encoder(feat)              # (B, L, 512)
        # TransformerEncoder expects (L, B, D)
        feat = feat.permute(1, 0, 2)               # (L, B, 512)
        # Transformer Encoder
        feat = self.transformer_encoder(feat)     # (L, B, 512)
        # 戻して (B, L, 512)
        feat = feat.permute(1, 0, 2)
        # 最終時刻の出力だけ取り出す or 平均を取る
        out = feat[:, -1, :]  # (B, 512) ←最終時刻
        # out = feat.mean(dim=1)  # (B, 512) ←平均
        out = self.head(out)  # (B, 12)

        return out
