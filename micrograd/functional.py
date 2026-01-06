"""
函数式API
提供无状态的函数接口，类似PyTorch的torch.nn.functional
"""

from .tensor import Tensor
import math


# ==================== 激活函数 ====================

def relu(x):
    """
    ReLU激活函数
    
    Args:
        x: 输入张量
    
    Returns:
        激活后的张量
    
    Example:
        >>> x = tensor([-1, 0, 1, 2])
        >>> relu(x)
        tensor([0.0000, 0.0000, 1.0000, 2.0000])
    """
    return x.relu()


def sigmoid(x):
    """
    Sigmoid激活函数
    
    Args:
        x: 输入张量
    
    Returns:
        激活后的张量，值域为(0, 1)
    
    Example:
        >>> x = tensor([0, 1, 2])
        >>> sigmoid(x)
        tensor([0.5000, 0.7311, 0.8808])
    """
    return x.sigmoid()


def tanh(x):
    """
    Tanh激活函数
    
    Args:
        x: 输入张量
    
    Returns:
        激活后的张量，值域为(-1, 1)
    
    Example:
        >>> x = tensor([-1, 0, 1])
        >>> tanh(x)
        tensor([-0.7616, 0.0000, 0.7616])
    """
    return x.tanh()


def leaky_relu(x, negative_slope=0.01):
    """
    Leaky ReLU激活函数
    
    Args:
        x: 输入张量
        negative_slope: 负数部分的斜率
    
    Returns:
        激活后的张量
    
    Example:
        >>> x = tensor([-2, -1, 0, 1, 2])
        >>> leaky_relu(x, 0.1)
        tensor([-0.2000, -0.1000, 0.0000, 1.0000, 2.0000])
    """
    out_data = [max(0.0, val) + negative_slope * min(0.0, val) for val in x.data]
    
    out = Tensor.__new__(Tensor)
    out.data = out_data
    out.shape = x.shape
    out.ndim = x.ndim
    out.size = x.size
    out.requires_grad = x.requires_grad
    out.grad = None
    out._prev = {x}
    out._op = f'LeakyReLU({negative_slope})'
    
    if x.requires_grad:
        def _backward():
            if x.grad is None:
                x.grad = [0.0] * x.size
            for i in range(x.size):
                slope = 1.0 if x.data[i] > 0 else negative_slope
                x.grad[i] += out.grad[i] * slope
        out._backward = _backward
    
    return out


def softmax(x, dim=-1):
    """
    Softmax函数
    
    Args:
        x: 输入张量
        dim: 应用softmax的维度
    
    Returns:
        归一化后的张量，沿指定维度和为1
    
    Example:
        >>> x = tensor([[1, 2, 3], [4, 5, 6]])
        >>> softmax(x, dim=1)
        tensor([[0.0900, 0.2447, 0.6652],
                [0.0900, 0.2447, 0.6652]])
    """
    if x.ndim == 0:
        return Tensor(1.0, requires_grad=x.requires_grad)
    
    if dim < 0:
        dim = x.ndim + dim
    
    # 数值稳定性：减去最大值
    max_vals = []
    
    # 计算每个切片的最大值
    def get_max_along_dim():
        result = {}
        for i in range(x.size):
            indices = []
            temp = i
            for d in reversed(x.shape):
                indices.insert(0, temp % d)
                temp //= d
            
            key = tuple(indices[:dim] + indices[dim+1:])
            if key not in result:
                result[key] = x.data[i]
            else:
                result[key] = max(result[key], x.data[i])
        return result
    
    max_dict = get_max_along_dim()
    
    # 计算exp(x - max)
    exp_data = []
    for i in range(x.size):
        indices = []
        temp = i
        for d in reversed(x.shape):
            indices.insert(0, temp % d)
            temp //= d
        
        key = tuple(indices[:dim] + indices[dim+1:])
        exp_data.append(math.exp(x.data[i] - max_dict[key]))
    
    # 计算sum(exp)
    sum_dict = {}
    for i in range(x.size):
        indices = []
        temp = i
        for d in reversed(x.shape):
            indices.insert(0, temp % d)
            temp //= d
        
        key = tuple(indices[:dim] + indices[dim+1:])
        if key not in sum_dict:
            sum_dict[key] = exp_data[i]
        else:
            sum_dict[key] += exp_data[i]
    
    # 归一化
    out_data = []
    for i in range(x.size):
        indices = []
        temp = i
        for d in reversed(x.shape):
            indices.insert(0, temp % d)
            temp //= d
        
        key = tuple(indices[:dim] + indices[dim+1:])
        out_data.append(exp_data[i] / sum_dict[key])
    
    out = Tensor.__new__(Tensor)
    out.data = out_data
    out.shape = x.shape
    out.ndim = x.ndim
    out.size = x.size
    out.requires_grad = x.requires_grad
    out.grad = None
    out._prev = {x}
    out._op = f'Softmax(dim={dim})'
    
    if x.requires_grad:
        def _backward():
            if x.grad is None:
                x.grad = [0.0] * x.size
            
            # Softmax的梯度: s_i * (δ_ij - s_j)
            for i in range(x.size):
                indices_i = []
                temp = i
                for d in reversed(x.shape):
                    indices_i.insert(0, temp % d)
                    temp //= d
                
                grad_sum = 0.0
                for j in range(x.size):
                    indices_j = []
                    temp = j
                    for d in reversed(x.shape):
                        indices_j.insert(0, temp % d)
                        temp //= d
                    
                    # 检查是否在同一个切片
                    same_slice = all(
                        indices_i[k] == indices_j[k] 
                        for k in range(x.ndim) if k != dim
                    )
                    
                    if same_slice:
                        if i == j:
                            grad_sum += out.grad[j] * out.data[i] * (1 - out.data[j])
                        else:
                            grad_sum += out.grad[j] * out.data[i] * (-out.data[j])
                
                x.grad[i] += grad_sum
        
        out._backward = _backward
    
    return out


def log_softmax(x, dim=-1):
    """
    Log-Softmax函数（数值稳定版本）
    
    Args:
        x: 输入张量
        dim: 应用log_softmax的维度
    
    Returns:
        log(softmax(x))
    
    Example:
        >>> x = tensor([[1, 2, 3]])
        >>> log_softmax(x, dim=1)
    """
    return softmax(x, dim=dim).log()


# ==================== 损失函数 ====================

def mse_loss(pred, target, reduction='mean'):
    """
    均方误差损失
    
    Args:
        pred: 预测值
        target: 目标值
        reduction: 'mean', 'sum' 或 'none'
    
    Returns:
        损失值
    
    Example:
        >>> pred = tensor([1, 2, 3])
        >>> target = tensor([1, 2, 2])
        >>> mse_loss(pred, target)
        tensor(0.3333)
    """
    diff = pred - target
    loss = diff * diff
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def binary_cross_entropy(pred, target, reduction='mean'):
    """
    二元交叉熵损失
    
    Args:
        pred: 预测值（0-1之间）
        target: 目标值（0或1）
        reduction: 'mean', 'sum' 或 'none'
    
    Returns:
        损失值
    
    Example:
        >>> pred = tensor([0.8, 0.3, 0.6])
        >>> target = tensor([1, 0, 1])
        >>> binary_cross_entropy(pred, target)
    """
    eps = 1e-7
    
    # 数值稳定性：clip预测值
    pred_clipped_data = [max(eps, min(1 - eps, p)) for p in pred.data]
    pred_clipped = Tensor.__new__(Tensor)
    pred_clipped.data = pred_clipped_data
    pred_clipped.shape = pred.shape
    pred_clipped.ndim = pred.ndim
    pred_clipped.size = pred.size
    pred_clipped.requires_grad = pred.requires_grad
    pred_clipped.grad = None
    pred_clipped._prev = {pred}
    pred_clipped._op = 'clip'
    
    if pred.requires_grad:
        def _backward():
            if pred.grad is None:
                pred.grad = [0.0] * pred.size
            for i in range(pred.size):
                # 只有在eps和1-eps之间的值才传递梯度
                if eps < pred.data[i] < 1 - eps:
                    pred.grad[i] += pred_clipped.grad[i]
        pred_clipped._backward = _backward
    
    # -[y*log(p) + (1-y)*log(1-p)]
    loss = -(target * pred_clipped.log() + (1 - target) * (1 - pred_clipped).log())
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def cross_entropy(pred, target, reduction='mean'):
    """
    交叉熵损失（多分类）
    
    Args:
        pred: 预测logits，形状 (N, C)
        target: 目标类别索引，形状 (N,) 或 one-hot形式 (N, C)
        reduction: 'mean', 'sum' 或 'none'
    
    Returns:
        损失值
    
    Example:
        >>> pred = tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
        >>> target = tensor([0, 1])  # 类别索引
        >>> cross_entropy(pred, target)
    """
    # 计算log_softmax
    log_probs = log_softmax(pred, dim=-1)
    
    # 如果target是类别索引
    if target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1):
        # 提取对应类别的log概率
        batch_size = pred.shape[0]
        num_classes = pred.shape[1]
        
        loss_data = []
        for i in range(batch_size):
            target_class = int(target.data[i])
            loss_data.append(-log_probs.data[i * num_classes + target_class])
        
        loss = Tensor.__new__(Tensor)
        loss.data = loss_data
        loss.shape = (batch_size,)
        loss.ndim = 1
        loss.size = batch_size
        loss.requires_grad = pred.requires_grad
        loss.grad = None
        loss._prev = {log_probs, target}
        loss._op = 'CrossEntropy'
        
        if pred.requires_grad:
            def _backward():
                if log_probs.grad is None:
                    log_probs.grad = [0.0] * log_probs.size
                
                for i in range(batch_size):
                    target_class = int(target.data[i])
                    log_probs.grad[i * num_classes + target_class] -= loss.grad[i]
            
            loss._backward = _backward
    else:
        # target是one-hot形式
        loss = -(target * log_probs)
        loss = loss.sum(dim=-1)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def nll_loss(log_probs, target, reduction='mean'):
    """
    负对数似然损失
    
    Args:
        log_probs: 对数概率，形状 (N, C)
        target: 目标类别索引，形状 (N,)
        reduction: 'mean', 'sum' 或 'none'
    
    Returns:
        损失值
    """
    batch_size = log_probs.shape[0]
    num_classes = log_probs.shape[1]
    
    loss_data = []
    for i in range(batch_size):
        target_class = int(target.data[i])
        loss_data.append(-log_probs.data[i * num_classes + target_class])
    
    loss = Tensor.__new__(Tensor)
    loss.data = loss_data
    loss.shape = (batch_size,)
    loss.ndim = 1
    loss.size = batch_size
    loss.requires_grad = log_probs.requires_grad
    loss.grad = None
    loss._prev = {log_probs, target}
    loss._op = 'NLLLoss'
    
    if log_probs.requires_grad:
        def _backward():
            if log_probs.grad is None:
                log_probs.grad = [0.0] * log_probs.size
            
            for i in range(batch_size):
                target_class = int(target.data[i])
                log_probs.grad[i * num_classes + target_class] -= loss.grad[i]
        
        loss._backward = _backward
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


# ==================== 其他操作 ====================

def dropout(x, p=0.5, training=True):
    """
    Dropout正则化
    
    Args:
        x: 输入张量
        p: dropout概率
        training: 是否在训练模式
    
    Returns:
        应用dropout后的张量
    
    Example:
        >>> x = tensor([1, 2, 3, 4])
        >>> dropout(x, p=0.5, training=True)
    """
    if not training or p == 0:
        return x
    
    import random
    
    # 生成mask并缩放
    scale = 1.0 / (1.0 - p)
    out_data = []
    mask = []
    
    for val in x.data:
        if random.random() > p:
            out_data.append(val * scale)
            mask.append(scale)
        else:
            out_data.append(0.0)
            mask.append(0.0)
    
    out = Tensor.__new__(Tensor)
    out.data = out_data
    out.shape = x.shape
    out.ndim = x.ndim
    out.size = x.size
    out.requires_grad = x.requires_grad
    out.grad = None
    out._prev = {x}
    out._op = f'Dropout(p={p})'
    
    if x.requires_grad:
        def _backward():
            if x.grad is None:
                x.grad = [0.0] * x.size
            for i in range(x.size):
                x.grad[i] += out.grad[i] * mask[i]
        out._backward = _backward
    
    return out


def linear(x, weight, bias=None):
    """
    线性变换: y = xW^T + b
    
    Args:
        x: 输入张量，形状 (*, in_features)
        weight: 权重张量，形状 (out_features, in_features)
        bias: 偏置张量，形状 (out_features,)，可选
    
    Returns:
        输出张量，形状 (*, out_features)
    
    Example:
        >>> x = tensor([[1, 2], [3, 4]])
        >>> weight = tensor([[1, 1], [2, 2]])
        >>> linear(x, weight)
    """
    # x @ weight.T
    out = x @ weight.reshape(weight.shape[1], weight.shape[0])
    
    if bias is not None:
        out = out + bias
    
    return out


def batch_norm_1d(x, running_mean, running_var, weight=None, bias=None, 
                  training=True, momentum=0.1, eps=1e-5):
    """
    一维批归一化
    
    Args:
        x: 输入张量，形状 (N, C)
        running_mean: 运行时均值
        running_var: 运行时方差
        weight: 缩放参数
        bias: 偏移参数
        training: 是否训练模式
        momentum: 动量
        eps: 数值稳定性常数
    
    Returns:
        归一化后的张量
    """
    if training:
        # 计算批次统计量
        mean = x.mean(dim=0, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=0, keepdim=True)
        
        # 更新运行时统计量
        for i in range(len(running_mean.data)):
            running_mean.data[i] = (1 - momentum) * running_mean.data[i] + momentum * mean.data[i]
            running_var.data[i] = (1 - momentum) * running_var.data[i] + momentum * var.data[i]
    else:
        mean = running_mean
        var = running_var
    
    # 归一化
    x_norm = (x - mean) / ((var + eps) ** 0.5)
    
    # 缩放和偏移
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    
    return x_norm


def pad(x, pad_width, mode='constant', value=0):
    """
    填充张量
    
    Args:
        x: 输入张量
        pad_width: 填充宽度，格式为 (left, right) 或 ((top, bottom), (left, right))
        mode: 填充模式，目前只支持 'constant'
        value: 填充值
    
    Returns:
        填充后的张量
    
    Example:
        >>> x = tensor([[1, 2], [3, 4]])
        >>> pad(x, (1, 1), value=0)  # 左右各填充1列
    """
    if mode != 'constant':
        raise NotImplementedError("Only constant padding is supported")
    
    # 简化版本：只支持2D张量的左右填充
    if x.ndim != 2:
        raise NotImplementedError("Only 2D tensors are supported")
    
    if isinstance(pad_width, int):
        pad_left = pad_right = pad_width
    else:
        pad_left, pad_right = pad_width
    
    rows, cols = x.shape
    new_cols = cols + pad_left + pad_right
    
    out_data = []
    for i in range(rows):
        # 左填充
        out_data.extend([value] * pad_left)
        # 原始数据
        out_data.extend(x.data[i * cols:(i + 1) * cols])
        # 右填充
        out_data.extend([value] * pad_right)
    
    out = Tensor.__new__(Tensor)
    out.data = out_data
    out.shape = (rows, new_cols)
    out.ndim = 2
    out.size = len(out_data)
    out.requires_grad = x.requires_grad
    out.grad = None
    out._prev = {x}
    out._op = f'Pad({pad_width})'
    
    if x.requires_grad:
        def _backward():
            if x.grad is None:
                x.grad = [0.0] * x.size
            
            for i in range(rows):
                for j in range(cols):
                    out_idx = i * new_cols + pad_left + j
                    in_idx = i * cols + j
                    x.grad[in_idx] += out.grad[out_idx]
        
        out._backward = _backward
    
    return out


def flatten(x, start_dim=0, end_dim=-1):
    """
    展平张量
    
    Args:
        x: 输入张量
        start_dim: 开始展平的维度
        end_dim: 结束展平的维度
    
    Returns:
        展平后的张量
    
    Example:
        >>> x = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
        >>> flatten(x, start_dim=1)  # (2, 4)
    """
    if end_dim < 0:
        end_dim = x.ndim + end_dim
    
    # 计算新形状
    new_shape = list(x.shape[:start_dim])
    
    flatten_size = 1
    for i in range(start_dim, end_dim + 1):
        flatten_size *= x.shape[i]
    new_shape.append(flatten_size)
    
    new_shape.extend(x.shape[end_dim + 1:])
    
    return x.reshape(tuple(new_shape))
