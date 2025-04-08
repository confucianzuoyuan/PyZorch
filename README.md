# 笔记

## 矩阵求导

$Y=WX$ ，其中 $W$ 是 $m \times n$ 矩阵，$X$ 是 $n \times p$ 矩阵。$Y$ 是 $m \times p$ 矩阵。

则求导结果如下：

$$
\frac{\partial Y}{\partial W} = X^T
$$

以及

$$
\frac{\partial Y}{\partial X} = W^T
$$

推导过程：

由于我们最终是要计算损失函数的值，而这个值是 **标量** 。

$$
z = L(Y)
$$

由于

$$
Y_{ij} = \sum_{k=1}^{n}W_{ik}X_{kj}
$$

> 多元函数链式法则
> 
> $$
> w = f(x, y) \\
> x = x(t) \\
> y = y(t)
> $$
> 
> 则
> 
> $$
> \frac{dw}{dt} = \frac{df}{dx}\frac{dx}{dt} + \frac{df}{dy}\frac{dy}{dt}
> $$
> 
> 也就是说 $t$ 的微小变化，导致了 $x$ 和 $y$ 的微小变化。从而导致了 $w$ 的微小变化，这两部分的微小变化需要相加，就是 $w$ 的微小变化。

所以，根据多元函数链式法则，有以下：

$$
\frac{dL}{dW_{dc}} = \sum_{i=1}^{m}\sum_{j=1}^{q}\frac{\partial L}{\partial Y_{ij}}\frac{\partial Y_{ij}}{\partial W_{dc}}
$$

而当 $i \neq d$ 时，

$$
\frac{\partial Y_{ij}}{\partial W_{dc}} = 0
$$

所以

$$
\frac{\partial L}{\partial W_{dc}} = \sum_{j=1}^{q}\frac{\partial L}{\partial Y_{dj}}\frac{\partial Y_{dj}}{\partial W_{dc}}
$$

而又因为

$$
\frac{\partial Y_{dj}}{\partial W_{dc}} = X_{cj}
$$

所以

$$
\frac{\partial L}{\partial W_{dc}} = \sum_{j=1}^{q}\frac{\partial L}{\partial Y_{dj}}X_{cj}
$$

所以

$$
\frac{dL}{dW_{dc}} = \sum_{j=1}^{q}\frac{\partial L}{\partial Y_{dj}}X_{jc}^T
$$

所以

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y}X^T
$$

所以

$$
\frac{\partial Y}{\partial W} = X^T
$$

## 广播语义

如果 PyTorch 操作支持广播，则可以自动扩展其张量参数，使其大小相等（而无需复制数据）。

### 通用语义

如果满足以下规则，则两个张量是“可广播的”

- 每个张量至少有一个维度。
- 当迭代维度大小时，从尾部维度开始，维度大小必须相等，其中一个为 1，或者其中一个不存在。

例如

```sh
>>> x=torch.empty(5,7,3)
>>> y=torch.empty(5,7,3)
# 相同形状的张量可广播，但不需要

>>> x=torch.empty((0,))
>>> y=torch.empty(2,2)
# x 和 y 不可以广播，因为 x 不满足条件1：也就是每个张量至少有一个维度。

# 将尾部的维度对齐
>>> x=torch.empty(5,3,4,1)
>>> y=torch.empty(  3,1,1)
# x and y are broadcastable.
# x 和 y 可以广播
# 逆序遍历：
# 最后一个都是 1，满足条件2
# 倒数第二个的 y 的维度是 1，满足条件2
# 倒数第三个的 x 和 y 的维度相同满足条件2
# 倒数第四个的 y 的维度不存在，满足条件2

# but:
>>> x=torch.empty(5,2,4,1)
>>> y=torch.empty(  3,1,1)
# x 和 y 不可广播，因为倒数第三个的维度 2 != 3，不满足条件2
```

如果两个张量 x、y 是“可广播的”，则结果张量大小的计算方式如下

- 如果 x 和 y 的维度数不相等，则在维度较少的张量的维度前添加 1，使它们的长度相等。
- 然后，对于每个维度大小，结果维度大小是 x 和 y 沿该维度的最大大小。

例如

```sh
# 维度的尾部对齐
>>> x=torch.empty(5,1,4,1)
>>> y=torch.empty(  3,1,1)
>>> (x+y).size()
torch.Size([5, 3, 4, 1])

>>> x=torch.empty(1)
>>> y=torch.empty(3,1,7)
>>> (x+y).size()
torch.Size([3, 1, 7])

>>> x=torch.empty(5,2,4,1)
>>> y=torch.empty(3,1,1)
>>> (x+y).size()
RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
```

## 算子的反向传播

### 加法

$$
L = f(x + y)
$$

$$
z = x + y
$$

$$
\frac{\partial L}{\partial x} = \frac{\partial f}{\partial z} \times \frac{\partial z}{\partial x} = \frac{\partial f}{\partial z} \times 1
$$

$$
\frac{\partial L}{\partial y} = \frac{\partial f}{\partial z} \times \frac{\partial z}{\partial y} = \frac{\partial f}{\partial z} \times 1
$$

$gradient = \partial f / \partial z$ 是反向传播过来的梯度，所以，加法的反向传播是：

$$
(gradient, gradient)
$$

### 减法

$$
L = f(x - y)
$$

$$
z = x - y
$$

$$
\frac{\partial L}{\partial x} = \frac{\partial f}{\partial z} \times \frac{\partial z}{\partial x} = \frac{\partial f}{\partial z} \times 1
$$

$$
\frac{\partial L}{\partial y} = -\frac{\partial f}{\partial z} \times \frac{\partial z}{\partial y} = -\frac{\partial f}{\partial z} \times 1
$$

$gradient = \partial f / \partial z$ 是反向传播过来的梯度，所以，加法的反向传播是：

$$
(gradient, -gradient)
$$

### 逐点乘法（element-wise mul）

$$
L = f(x \odot y)
$$

$$
z = x \odot y
$$

$$
\frac{\partial L}{\partial x} = \frac{\partial f}{\partial z} \odot \frac{\partial z}{\partial x} = \frac{\partial f}{\partial z} \odot y
$$

$$
\frac{\partial L}{\partial y} = \frac{\partial f}{\partial z} \odot \frac{\partial z}{\partial y} = \frac{\partial f}{\partial z} \odot x
$$

$gradient = \partial f / \partial z$ 是反向传播过来的梯度，所以，加法的反向传播是：

$$
(gradient \odot y, gradient \odot x)
$$