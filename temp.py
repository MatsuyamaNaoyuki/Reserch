import torch

print("CUDAが使えますか？:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("使っているGPUの名前:", torch.cuda.get_device_name(0))
else:
    print("GPUが見つかりませんでした。")
