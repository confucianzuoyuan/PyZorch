"""
优化器
"""


class Optimizer:
    """优化器基类"""
    
    def __init__(self, parameters):
        self.parameters = list(parameters)
    
    def zero_grad(self):
        """清零梯度"""
        for p in self.parameters:
            p.zero_grad()
    
    def step(self):
        """更新参数"""
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降"""
    
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [[0.0] * p.size for p in self.parameters]
    
    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            for j in range(p.size):
                # 动量更新
                self.velocities[i][j] = (self.momentum * self.velocities[i][j] - 
                                        self.lr * p.grad[j])
                p.data[j] += self.velocities[i][j]


class Adam(Optimizer):
    """Adam优化器"""
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        self.m = [[0.0] * p.size for p in self.parameters]
        self.v = [[0.0] * p.size for p in self.parameters]
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            for j in range(p.size):
                # 更新一阶矩估计
                self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * p.grad[j]
                
                # 更新二阶矩估计
                self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * (p.grad[j] ** 2)
                
                # 偏差修正
                m_hat = self.m[i][j] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][j] / (1 - self.beta2 ** self.t)
                
                # 更新参数
                p.data[j] -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
