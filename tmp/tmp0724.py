import torch
import torch.nn as nn

# 创建一个示例张量
x = torch.randn(10, 20)

# 定义一个LayerNorm层
layer_norm = nn.LayerNorm(x.size()[1:])

# 对张量进行LayerNorm处理
output = layer_norm(x)

print(output.shape)  # 输出: torch.Size([10, 20])
