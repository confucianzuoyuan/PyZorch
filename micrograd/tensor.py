"""
MicroGrad - 纯Python深度学习框架
核心张量类，支持自动微分
"""

class Tensor:
    """
    张量类，支持自动微分的多维数组
    """
    
    def __init__(self, data, requires_grad=False, _children=(), _op='', _label=''):
        """
        初始化张量
        
        Args:
            data: 数值、列表或嵌套列表
            requires_grad: 是否需要计算梯度
            _children: 父节点（用于构建计算图）
            _op: 创建此张量的操作
            _label: 张量标签（用于调试）
        """
        # 解析数据和形状
        self.data, self.shape = self._parse_data(data)
        self.ndim = len(self.shape)
        self.size = len(self.data)
        
        # 梯度相关
        self.requires_grad = requires_grad
        self.grad = None
        
        # 计算图相关
        self._prev = set(_children)
        self._op = _op
        self._label = _label
        self._backward = lambda: None  # 默认的空反向传播函数
        
    @staticmethod
    def _parse_data(data):
        """解析数据，返回扁平化数据和形状"""
        if isinstance(data, (int, float)):
            return [float(data)], ()
        
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Cannot create tensor from empty list")
            
            # 检查是否为嵌套列表
            if isinstance(data[0], (int, float)):
                return [float(x) for x in data], (len(data),)
            elif isinstance(data[0], list):
                flat_data = []
                first_inner, inner_shape = Tensor._parse_data(data[0])
                flat_data.extend(first_inner)
                
                for i, item in enumerate(data[1:], 1):
                    inner_data, item_shape = Tensor._parse_data(item)
                    if item_shape != inner_shape:
                        raise ValueError(f"Inconsistent shape at index {i}")
                    flat_data.extend(inner_data)
                
                return flat_data, (len(data),) + inner_shape
            else:
                raise TypeError(f"Unsupported type: {type(data[0])}")
        
        raise TypeError(f"Unsupported type: {type(data)}")
    
    @classmethod
    def _create_tensor(cls, data, shape, requires_grad=False, _children=(), _op=''):
        """
        内部方法：创建张量（用于操作结果）
        确保所有属性都被正确初始化
        """
        out = cls.__new__(cls)
        out.data = data
        out.shape = shape
        out.ndim = len(shape)
        out.size = len(data)
        out.requires_grad = requires_grad
        out.grad = None
        out._prev = set(_children)
        out._op = _op
        out._label = ''
        out._backward = lambda: None  # 重要：默认空函数
        return out
    
    def reshape(self, *new_shape):
        """重塑张量形状"""
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = tuple(new_shape[0])
        
        # 计算新形状的总大小
        new_size = 1
        for dim in new_shape:
            new_size *= dim
        
        if new_size != self.size:
            raise ValueError(f"Cannot reshape tensor of size {self.size} to {new_shape}")
        
        out = Tensor._create_tensor(
            self.data.copy(),
            new_shape,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='reshape'
        )
        
        if self.requires_grad:
            def _backward():
                if self.grad is None:
                    self.grad = [0.0] * self.size
                for i in range(self.size):
                    self.grad[i] += out.grad[i]
            out._backward = _backward
        
        return out
    
    def _get_index(self, indices):
        """将多维索引转换为扁平索引"""
        if self.ndim == 0:
            return 0
        
        if not isinstance(indices, tuple):
            indices = (indices,)
        
        # 计算步幅
        strides = [1]
        for i in range(self.ndim - 1, 0, -1):
            strides.insert(0, strides[0] * self.shape[i])
        
        index = 0
        for i, idx in enumerate(indices):
            if idx < 0:
                idx = self.shape[i] + idx
            if idx < 0 or idx >= self.shape[i]:
                raise IndexError(f"Index {idx} out of bounds for dimension {i}")
            index += idx * strides[i]
        
        return index
    
    def __getitem__(self, indices):
        """获取元素"""
        if self.ndim == 0:
            return self.data[0]
        
        index = self._get_index(indices)
        return self.data[index]
    
    def __setitem__(self, indices, value):
        """设置元素"""
        if self.ndim == 0:
            self.data[0] = float(value)
        else:
            index = self._get_index(indices)
            self.data[index] = float(value)
    
    # ==================== 算术运算 ====================
    
    def __add__(self, other):
        """加法"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # 广播检查
        out_shape, self_broadcast, other_broadcast = self._broadcast_shapes(self.shape, other.shape)
        out_data = [0.0] * self._shape_size(out_shape)
        
        # 执行加法
        for i in range(len(out_data)):
            self_idx = self._broadcast_index(i, out_shape, self.shape, self_broadcast)
            other_idx = self._broadcast_index(i, out_shape, other.shape, other_broadcast)
            out_data[i] = self.data[self_idx] + other.data[other_idx]
        
        out = Tensor._create_tensor(
            out_data,
            out_shape,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='+'
        )
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = [0.0] * self.size
                    self._accumulate_grad(out.grad, out_shape, self.shape, self_broadcast)
                
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = [0.0] * other.size
                    other._accumulate_grad(out.grad, out_shape, other.shape, other_broadcast)
            
            out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        """乘法"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out_shape, self_broadcast, other_broadcast = self._broadcast_shapes(self.shape, other.shape)
        out_data = [0.0] * self._shape_size(out_shape)
        
        for i in range(len(out_data)):
            self_idx = self._broadcast_index(i, out_shape, self.shape, self_broadcast)
            other_idx = self._broadcast_index(i, out_shape, other.shape, other_broadcast)
            out_data[i] = self.data[self_idx] * other.data[other_idx]
        
        out = Tensor._create_tensor(
            out_data,
            out_shape,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='*'
        )
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = [0.0] * self.size
                    # d(self * other)/d(self) = other
                    grad_contrib = [0.0] * len(out_data)
                    for i in range(len(out_data)):
                        other_idx = self._broadcast_index(i, out_shape, other.shape, other_broadcast)
                        grad_contrib[i] = out.grad[i] * other.data[other_idx]
                    self._accumulate_grad(grad_contrib, out_shape, self.shape, self_broadcast)
                
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = [0.0] * other.size
                    # d(self * other)/d(other) = self
                    grad_contrib = [0.0] * len(out_data)
                    for i in range(len(out_data)):
                        self_idx = self._broadcast_index(i, out_shape, self.shape, self_broadcast)
                        grad_contrib[i] = out.grad[i] * self.data[self_idx]
                    other._accumulate_grad(grad_contrib, out_shape, other.shape, other_broadcast)
            
            out._backward = _backward
        
        return out
    
    def __pow__(self, power):
        """幂运算"""
        assert isinstance(power, (int, float)), "Only support int/float power"
        
        out_data = [x ** power for x in self.data]
        
        out = Tensor._create_tensor(
            out_data,
            self.shape,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f'**{power}'
        )
        
        if self.requires_grad:
            def _backward():
                if self.grad is None:
                    self.grad = [0.0] * self.size
                # d(x^n)/dx = n * x^(n-1)
                for i in range(self.size):
                    self.grad[i] += out.grad[i] * power * (self.data[i] ** (power - 1))
            out._backward = _backward
        
        return out
    
    def __neg__(self):
        """取负"""
        return self * -1
    
    def __sub__(self, other):
        """减法"""
        return self + (-other)
    
    def __truediv__(self, other):
        """除法"""
        return self * (other ** -1)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    # ==================== 激活函数 ====================
    
    def relu(self):
        """ReLU激活函数"""
        out_data = [max(0.0, x) for x in self.data]
        
        out = Tensor._create_tensor(
            out_data,
            self.shape,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='ReLU'
        )
        
        if self.requires_grad:
            def _backward():
                if self.grad is None:
                    self.grad = [0.0] * self.size
                for i in range(self.size):
                    self.grad[i] += out.grad[i] * (1.0 if self.data[i] > 0 else 0.0)
            out._backward = _backward
        
        return out
    
    def sigmoid(self):
        """Sigmoid激活函数"""
        import math
        out_data = [1.0 / (1.0 + math.exp(-x)) for x in self.data]
        
        out = Tensor._create_tensor(
            out_data,
            self.shape,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='Sigmoid'
        )
        
        if self.requires_grad:
            def _backward():
                if self.grad is None:
                    self.grad = [0.0] * self.size
                for i in range(self.size):
                    s = out.data[i]
                    self.grad[i] += out.grad[i] * s * (1 - s)
            out._backward = _backward
        
        return out
    
    def tanh(self):
        """Tanh激活函数"""
        import math
        out_data = [math.tanh(x) for x in self.data]
        
        out = Tensor._create_tensor(
            out_data,
            self.shape,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='Tanh'
        )
        
        if self.requires_grad:
            def _backward():
                if self.grad is None:
                    self.grad = [0.0] * self.size
                for i in range(self.size):
                    t = out.data[i]
                    self.grad[i] += out.grad[i] * (1 - t * t)
            out._backward = _backward
        
        return out
    
    def exp(self):
        """指数函数"""
        import math
        out_data = [math.exp(x) for x in self.data]
        
        out = Tensor._create_tensor(
            out_data,
            self.shape,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='exp'
        )
        
        if self.requires_grad:
            def _backward():
                if self.grad is None:
                    self.grad = [0.0] * self.size
                for i in range(self.size):
                    self.grad[i] += out.grad[i] * out.data[i]
            out._backward = _backward
        
        return out
    
    def log(self):
        """自然对数"""
        import math
        out_data = [math.log(x) if x > 0 else float('-inf') for x in self.data]
        
        out = Tensor._create_tensor(
            out_data,
            self.shape,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='log'
        )
        
        if self.requires_grad:
            def _backward():
                if self.grad is None:
                    self.grad = [0.0] * self.size
                for i in range(self.size):
                    if self.data[i] > 0:
                        self.grad[i] += out.grad[i] / self.data[i]
            out._backward = _backward
        
        return out
    
    # ==================== 归约操作 ====================
    
    def sum(self, dim=None, keepdim=False):
        """求和"""
        if dim is None:
            # 全局求和
            total = sum(self.data)
            out = Tensor(total, requires_grad=self.requires_grad)
            out._prev = {self}
            out._op = 'sum'
            
            if self.requires_grad:
                def _backward():
                    if self.grad is None:
                        self.grad = [0.0] * self.size
                    for i in range(self.size):
                        self.grad[i] += out.grad[0]
                out._backward = _backward
            
            return out
        
        # 沿指定维度求和
        if dim < 0:
            dim = self.ndim + dim
        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"Invalid dim {dim} for tensor with {self.ndim} dimensions")
        
        # 计算输出形状
        if keepdim:
            out_shape = list(self.shape)
            out_shape[dim] = 1
            out_shape = tuple(out_shape)
        else:
            out_shape = self.shape[:dim] + self.shape[dim+1:]
        
        out_size = self._shape_size(out_shape)
        out_data = [0.0] * out_size
        
        # 执行求和
        for i in range(self.size):
            out_idx = self._reduce_index(i, dim, keepdim)
            out_data[out_idx] += self.data[i]
        
        out = Tensor._create_tensor(
            out_data,
            out_shape,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f'sum(dim={dim})'
        )
        
        if self.requires_grad:
            def _backward():
                if self.grad is None:
                    self.grad = [0.0] * self.size
                for i in range(self.size):
                    out_idx = self._reduce_index(i, dim, keepdim)
                    self.grad[i] += out.grad[out_idx]
            out._backward = _backward
        
        return out
    
    def mean(self, dim=None, keepdim=False):
        """求平均"""
        if dim is None:
            return self.sum() / self.size
        
        count = self.shape[dim]
        return self.sum(dim=dim, keepdim=keepdim) / count
    
    # ==================== 矩阵运算 ====================
    
    def matmul(self, other):
        """矩阵乘法"""
        if not isinstance(other, Tensor):
            raise TypeError("matmul requires Tensor")
        
        # 支持2D矩阵乘法
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul only supports 2D tensors")
        
        m, k1 = self.shape
        k2, n = other.shape
        
        if k1 != k2:
            raise ValueError(f"Shape mismatch: ({m}, {k1}) @ ({k2}, {n})")
        
        out_data = [0.0] * (m * n)
        
        # 执行矩阵乘法
        for i in range(m):
            for j in range(n):
                total = 0.0
                for k in range(k1):
                    total += self.data[i * k1 + k] * other.data[k * n + j]
                out_data[i * n + j] = total
        
        out = Tensor._create_tensor(
            out_data,
            (m, n),
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='@'
        )
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = [0.0] * self.size
                    # dL/dA = dL/dC @ B^T
                    for i in range(m):
                        for k in range(k1):
                            for j in range(n):
                                self.grad[i * k1 + k] += out.grad[i * n + j] * other.data[k * n + j]
                
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = [0.0] * other.size
                    # dL/dB = A^T @ dL/dC
                    for k in range(k1):
                        for j in range(n):
                            for i in range(m):
                                other.grad[k * n + j] += self.data[i * k1 + k] * out.grad[i * n + j]
            
            out._backward = _backward
        
        return out
    
    def __matmul__(self, other):
        return self.matmul(other)
    
    # ==================== 反向传播 ====================
    
    def backward(self):
        """反向传播"""
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on tensor that doesn't require grad")
        
        if self.size != 1:
            raise RuntimeError("backward should be called only on scalar")
        
        # 初始化梯度
        self.grad = [1.0]
        
        # 拓扑排序
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # 反向传播
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """清零梯度"""
        self.grad = None
    
    # ==================== 辅助方法 ====================
    
    @staticmethod
    def _shape_size(shape):
        """计算形状的总大小"""
        if len(shape) == 0:
            return 1
        size = 1
        for dim in shape:
            size *= dim
        return size
    
    @staticmethod
    def _broadcast_shapes(shape1, shape2):
        """计算广播后的形状"""
        # 对齐维度
        ndim = max(len(shape1), len(shape2))
        s1 = (1,) * (ndim - len(shape1)) + shape1
        s2 = (1,) * (ndim - len(shape2)) + shape2
        
        # 检查广播兼容性
        out_shape = []
        for d1, d2 in zip(s1, s2):
            if d1 == d2:
                out_shape.append(d1)
            elif d1 == 1:
                out_shape.append(d2)
            elif d2 == 1:
                out_shape.append(d1)
            else:
                raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")
        
        # 记录哪些维度需要广播
        broadcast1 = tuple(d1 == 1 and d2 != 1 for d1, d2 in zip(s1, s2))
        broadcast2 = tuple(d2 == 1 and d1 != 1 for d1, d2 in zip(s1, s2))
        
        return tuple(out_shape), broadcast1, broadcast2
    
    def _broadcast_index(self, flat_idx, out_shape, in_shape, broadcast_dims):
        """将输出索引映射到输入索引（考虑广播）"""
        # 计算输出的多维索引
        out_indices = []
        temp = flat_idx
        for dim in reversed(out_shape):
            out_indices.insert(0, temp % dim)
            temp //= dim
        
        # 对齐维度
        ndim_diff = len(out_shape) - len(in_shape)
        in_indices = out_indices[ndim_diff:]
        
        # 应用广播规则
        for i, (idx, broadcast) in enumerate(zip(in_indices, broadcast_dims[ndim_diff:])):
            if broadcast:
                in_indices[i] = 0
        
        # 转换为扁平索引
        flat_in_idx = 0
        stride = 1
        for i in reversed(range(len(in_shape))):
            flat_in_idx += in_indices[i] * stride
            stride *= in_shape[i]
        
        return flat_in_idx
    
    def _accumulate_grad(self, grad_data, out_shape, in_shape, broadcast_dims):
        """累积梯度（处理广播）"""
        for i in range(len(grad_data)):
            in_idx = self._broadcast_index(i, out_shape, in_shape, broadcast_dims)
            self.grad[in_idx] += grad_data[i]
    
    def _reduce_index(self, in_idx, dim, keepdim):
        """计算归约操作的输出索引"""
        # 计算输入的多维索引
        in_indices = []
        temp = in_idx
        for d in reversed(self.shape):
            in_indices.insert(0, temp % d)
            temp //= d
        
        # 移除或保持归约维度
        if keepdim:
            out_indices = in_indices[:dim] + [0] + in_indices[dim+1:]
            out_shape = self.shape[:dim] + (1,) + self.shape[dim+1:]
        else:
            out_indices = in_indices[:dim] + in_indices[dim+1:]
            out_shape = self.shape[:dim] + self.shape[dim+1:]
        
        # 转换为扁平索引
        out_idx = 0
        stride = 1
        for i in reversed(range(len(out_shape))):
            out_idx += out_indices[i] * stride
            stride *= out_shape[i]
        
        return out_idx
    
    # ==================== 字符串表示 ====================
    
    def __str__(self):
        if self.ndim == 0:
            grad_fn = f", grad_fn=<{self._op}>" if self._op else ""
            return f"tensor({self.data[0]:.4f}{grad_fn})"
        
        def format_array(indices, depth=0):
            if depth == self.ndim:
                return f"{self[indices]:.4f}"
            
            dim_size = self.shape[depth]
            if dim_size > 6:
                items = []
                for i in range(3):
                    items.append(format_array(indices + (i,), depth + 1))
                items.append('...')
                for i in range(dim_size - 3, dim_size):
                    items.append(format_array(indices + (i,), depth + 1))
            else:
                items = [format_array(indices + (i,), depth + 1) for i in range(dim_size)]
            
            if depth == self.ndim - 1:
                return '[' + ', '.join(items) + ']'
            else:
                indent = ' ' * (depth + 1)
                sep = ',\n' + indent
                return '[' + sep.join(items) + ']'
        
        tensor_str = format_array(())
        grad_fn = f", grad_fn=<{self._op}>" if self._op else ""
        return f"tensor({tensor_str}{grad_fn})"
    
    def __repr__(self):
        return self.__str__()


# ==================== 工厂函数 ====================

def tensor(data, requires_grad=False):
    """创建张量"""
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape, requires_grad=False):
    """创建全零张量"""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    
    def create_zeros(s):
        if len(s) == 0:
            return 0.0
        if len(s) == 1:
            return [0.0] * s[0]
        return [create_zeros(s[1:]) for _ in range(s[0])]
    
    data = create_zeros(shape) if len(shape) > 0 else 0.0
    return Tensor(data, requires_grad=requires_grad)


def ones(*shape, requires_grad=False):
    """创建全一张量"""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    
    def create_ones(s):
        if len(s) == 0:
            return 1.0
        if len(s) == 1:
            return [1.0] * s[0]
        return [create_ones(s[1:]) for _ in range(s[0])]
    
    data = create_ones(shape) if len(shape) > 0 else 1.0
    return Tensor(data, requires_grad=requires_grad)


def randn(*shape, requires_grad=False):
    """创建标准正态分布随机张量"""
    import random
    
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    
    def create_randn(s):
        if len(s) == 0:
            return random.gauss(0, 1)
        if len(s) == 1:
            return [random.gauss(0, 1) for _ in range(s[0])]
        return [create_randn(s[1:]) for _ in range(s[0])]
    
    data = create_randn(shape) if len(shape) > 0 else random.gauss(0, 1)
    return Tensor(data, requires_grad=requires_grad)


def rand(*shape, requires_grad=False):
    """创建均匀分布随机张量 [0, 1)"""
    import random
    
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    
    def create_rand(s):
        if len(s) == 0:
            return random.random()
        if len(s) == 1:
            return [random.random() for _ in range(s[0])]
        return [create_rand(s[1:]) for _ in range(s[0])]
    
    data = create_rand(shape) if len(shape) > 0 else random.random()
    return Tensor(data, requires_grad=requires_grad)
