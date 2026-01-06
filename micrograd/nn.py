"""
神经网络模块
"""

from .tensor import Tensor, zeros, randn
import math


class Module:
    """神经网络模块基类"""
    
    def zero_grad(self):
        """清零所有参数的梯度"""
        for p in self.parameters():
            p.zero_grad()
    
    def parameters(self):
        """返回所有参数"""
        return []


class Linear(Module):
    """全连接层"""
    
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier初始化
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = randn(out_features, in_features, requires_grad=True)
        for i in range(len(self.weight.data)):
            self.weight.data[i] *= std
        
        if bias:
            self.bias = zeros(out_features, requires_grad=True)
        else:
            self.bias = None
    
    def __call__(self, x):
        # x: (batch, in_features)
        # weight: (out_features, in_features)
        # output: (batch, out_features)
        out = x @ self.weight.reshape(self.in_features, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]


class ReLU(Module):
    """ReLU激活函数"""
    
    def __call__(self, x):
        return x.relu()


class Sigmoid(Module):
    """Sigmoid激活函数"""
    
    def __call__(self, x):
        return x.sigmoid()


class Tanh(Module):
    """Tanh激活函数"""
    
    def __call__(self, x):
        return x.tanh()


class Sequential(Module):
    """顺序容器"""
    
    def __init__(self, *layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class MSELoss(Module):
    """均方误差损失"""
    
    def __call__(self, pred, target):
        diff = pred - target
        return (diff * diff).mean()


class BCELoss(Module):
    """二元交叉熵损失"""
    
    def __call__(self, pred, target):
        # -[y*log(p) + (1-y)*log(1-p)]
        eps = 1e-7
        pred_clipped = pred  # 实际应该clip，这里简化
        loss = -(target * pred_clipped.log() + (1 - target) * (1 - pred_clipped).log())
        return loss.mean()
