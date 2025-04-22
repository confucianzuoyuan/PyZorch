# 在 PyTorch 中，sum 函数用于计算张量沿指定维度的元素总和。
# sum：该方法返回一个张量，其中包含沿指定维度的元素总和。其基本语法为：
# tensor.sum(dim=None, keepdim=False)
# - dim：指定要计算总和的维度。可以是单个整数或整数元组。如果不指定（即 dim=None），则计算所有元素的总和。
# - keepdim：布尔值，指示是否保留原始维度。默认为 False，如果设置为 True，则输出张量的维度将与输入张量相同，但被求和的维度将保持为 1。

import torch

# 创建一个二维张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 计算所有元素的总和
total_sum = tensor.sum()
print("所有元素的总和:", total_sum)

# 沿着第 0 维（行）计算总和
sum_dim0 = tensor.sum(dim=0)
print("沿着第 0 维的总和:", sum_dim0)

# 沿着第 1 维（列）计算总和
sum_dim1 = tensor.sum(dim=1)
print("沿着第 1 维的总和:", sum_dim1)

# 保持维度
sum_keepdim = tensor.sum(dim=0, keepdim=True)
print("沿着第 0 维的总和（保持维度）:\n", sum_keepdim)