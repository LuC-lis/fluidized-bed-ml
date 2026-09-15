import torch

# 打印关键信息
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")  # 仅CUDA版本存在
print(f"设备数量: {torch.cuda.device_count()}")
print(f"当前设备: {torch.cuda.current_device()}")
print(f"设备名称: {torch.cuda.get_device_name(0)}")  # 如果是NVIDIA显卡

# 测试简单计算
a = torch.randn(2, 3)
b = torch.randn(2, 3)
print("\n矩阵相加结果:")
print(a + b)
