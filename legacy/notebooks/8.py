import torch
import torch.nn as nn

# 打印版本
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())

# 创建一个简单的线性神经网络
model = nn.Sequential(
    nn.Linear(4, 8),    # 输入4维，输出8维
    nn.ReLU(),
    nn.Linear(8, 2)     # 输出2维
)

# 打印模型结构
print("\n模型结构:")
print(model)

# 构造一个输入张量
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # shape: (1, 4)

# 前向传播
output = model(x)

print("\n模型输出:")
print(output)
