"""
MicroGrad - 纯Python深度学习框架
"""

__version__ = '0.1.0'
__author__ = 'MicroGrad Team'

# 导入核心模块
from .tensor import (
    Tensor,
    tensor,
    zeros,
    ones,
    randn,
    rand,
)

# 导入神经网络模块
from .nn import (
    Module,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Sequential,
    MSELoss,
    BCELoss,
)

# 导入优化器
from .optim import (
    Optimizer,
    SGD,
    Adam,
)

# 导入函数式API
from . import functional as F

# 导入可视化工具
from .visualize import (
    draw_dot,
    draw_simple_dot,
    print_graph_info,
    trace,
)

# 定义公开接口
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    
    # 张量
    'Tensor',
    'tensor',
    'zeros',
    'ones',
    'randn',
    'rand',
    
    # 神经网络模块
    'Module',
    'Linear',
    'ReLU',
    'Sigmoid',
    'Tanh',
    'Sequential',
    'MSELoss',
    'BCELoss',
    
    # 优化器
    'Optimizer',
    'SGD',
    'Adam',
    
    # 函数式API
    'F',
    
    # 可视化
    'draw_dot',
    'draw_simple_dot',
    'print_graph_info',
    'trace',
]
