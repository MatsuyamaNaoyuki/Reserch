import torch

# 例のデータ
x = torch.tensor([
    [1.0, 2.0],
    [float('nan'), 3.0],
    [4.0, float('nan')],
    [5.0, 6.0]
])

# 各行に NaN が含まれるかどうか
nan_mask = torch.isnan(x).any(dim=1)

# 行番号（インデックス）のリスト
nan_rows = nan_mask.nonzero(as_tuple=True)[0].tolist()

print(nan_rows)