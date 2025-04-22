// Display inline code in a small box
// that retains the correct baseline.
#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt,
)

// Display block code in a larger block
// with more padding.
#show raw.where(block: true): block.with(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
)

= I 反向传播算法

所有算子的反向传播公式的推导，一定要从最后的损失函数的求值结果，也就是从#text(fill: red, "标量")开始进行反向传播。

\

== 1 矩阵乘法的反向传播

\

$Y = W X$，其中 $W$ 是 $m times n$ 矩阵，$X$ 是 $n times p$ 矩阵。那么 $Y$ 是 $m times p$ 矩阵。

假设损失函数是$L(Y)$，是一个标量。

那么结论是：

$ (partial L) / (partial W) = (partial L) / (partial Y) X^T $

以及

$ (partial L) / (partial X) = W^T (partial L) / (partial Y) $

将以上两个公式转换成代码就可以实现矩阵乘法的反向传播。因为反向传播过来的梯度是

$ "gradient" = (partial L) / (partial Y) $

推导过程

以第一个公式为例

因为

$ Y_(i j) = sum_(k=1)^n W_(i k)X_(k j)$

根据多元函数链式法则，有以下结论

$
(partial L) / (partial W_(d c))
=
sum_(i=1)^m sum_(j=1)^q
(partial L) / (partial Y_(i j))
(partial Y_(i j)) / (partial W_(d c))
$

而当 $i != d$ 时，

$
  (partial Y_(i j)) / (partial W_(d c)) = 0
$

所以

$
  (partial L) / (partial W_(d c))
  =
  sum_(j=1)^q
  (partial L) / (partial Y_(d j))
  (partial Y_(d j)) / (partial W_(d c))
$

而又因为

$
  (partial Y_(d j)) / (partial W_(d c)) = X_(c j)
$

所以

$
  (partial L) / (partial W_(d c))
  =
  sum_(j=1)^q
  (partial L) / (partial Y_(d j))
  X_(c j)
  =
  sum_(j=1)^q
  (partial L) / (partial Y_(d j))
  X_(j c)^T
$

所以

$
  (partial L) / (partial W) = (partial L) / (partial Y) X^T
$

另一个公式同理。

== 2 矩阵加法的反向传播

假设$Z=X+Y$，那么$Z_(i j)=X_(i j)+Y_(i j)$

损失函数是$L(Z)$

继续使用多元链式法则

$
  (partial L) / (partial X_(d c))
  =
  sum_i sum_j
  (partial L) / (partial Z_(i j))
  (partial Z_(i j)) / (partial X_(d c))
  =
  (partial L) / (partial Z_(d c))
  (partial Z_(d c)) / (partial X_(d c))
  =
  (partial L) / (partial Z_(d c))
$

所以

$
  (partial L) / (partial X) = (partial L) / (partial Z) = "gradient" \
  (partial L) / (partial Y) = (partial L) / (partial Z) = "gradient"
$

== 3 矩阵减法的反向传播

假设$Z=X+Y$，那么$Z_(i j)=X_(i j)-Y_(i j)$

沿用上一节的推导过程，可以得到

$
  (partial L) / (partial X) = (partial L) / (partial Z) = "gradient" \
  (partial L) / (partial Y) = (partial L) / (partial Z) = -"gradient"
$

== 4 矩阵的逐点乘法（哈达玛积）的反向传播

假设$Z=X dot.circle Y$，那么$Z_(i j)=X_(i j) times Y_(i j)$

损失函数是$L(Z)$

继续使用多元链式法则

$
  (partial L) / (partial X_(d c))
  =
  sum_i sum_j
  (partial L) / (partial Z_(i j))
  (partial Z_(i j)) / (partial X_(d c))
  =
  (partial L) / (partial Z_(d c))
  (partial Z_(d c)) / (partial X_(d c))
  =
  (partial L) / (partial Z_(d c))
  Y_(d c)
$

所以

$
  (partial L) / (partial X) = (partial L) / (partial Z) dot.circle Y = "gradient" dot.circle Y \
  (partial L) / (partial Y) = (partial L) / (partial Z) dot.circle X = "gradient" dot.circle X
$

== 5 对矩阵进行逐点运算的反向传播

=== $sin$函数的反向传播

假设 $Y=sin(X)$，那么含义如下：

$
  Y_(i j) = sin (X_(i j))
$

假设损失函数是 $L(Y)$

继续使用多元链式法则

$
  (partial L) / (partial X_(d c))
  =
  sum_i sum_j
  (partial L) / (partial Y_(i j))
  (partial Y_(i j)) / (partial X_(d c))
  =
  (partial L) / (partial Y_(d c))
  (partial Y_(d c)) / (partial X_(d c))
  =
  (partial L) / (partial Y_(d c))
  cos (X_(d c))
$

所以有

$
  (partial L) / (partial X) = (partial L) / (partial Z) dot.circle cos(X) = "gradient" dot.circle cos(X)
$

=== $cos$函数的反向传播

假设 $Y=sin(X)$，那么含义如下：

$
  Y_(i j) = cos (X_(i j))
$

假设损失函数是 $L(Y)$

继续使用多元链式法则

$
  (partial L) / (partial X_(d c))
  =
  sum_i sum_j
  (partial L) / (partial Y_(i j))
  (partial Y_(i j)) / (partial X_(d c))
  =
  (partial L) / (partial Y_(d c))
  (partial Y_(d c)) / (partial X_(d c))
  =
  -
  (partial L) / (partial Y_(d c))
  sin (X_(d c))
$

所以有

$
  (partial L) / (partial X) = -(partial L) / (partial Z) dot.circle sin (X) = -"gradient" dot.circle sin(X)
$

=== $log$函数的反向传播

假设 $Y=log(X)$，那么含义如下：

$
  Y_(i j) = log_e (X_(i j))
$

假设损失函数是 $L(Y)$

继续使用多元链式法则

$
  (partial L) / (partial X_(d c))
  =
  sum_i sum_j
  (partial L) / (partial Y_(i j))
  (partial Y_(i j)) / (partial X_(d c))
  =
  (partial L) / (partial Y_(d c))
  (partial Y_(d c)) / (partial X_(d c))
  =
  (partial L) / (partial Y_(d c))
  1 / (X_(d c))
$

所以有

$
  (partial L) / (partial X) = (partial L) / (partial Z) dot.circle 1 / (X) = "gradient" dot.circle 1 / X
$

= II 各种算子的语义

== 1 reshape

在 PyTorch 中，`reshape` 函数用于改变张量的形状（即维度），而不改变其数据。

```py
tensor.reshape(new_shape)
```

`new_shape`：一个整数或整数元组，表示新的维度。新形状的元素总数必须与原始张量的元素总数相同。

- reshape：该方法返回一个新张量，其数据与原始张量相同，但形状不同。
- 视图（View）：在大多数情况下，reshape 返回的是原始张量的视图（即共享相同的数据内存），这意味着更改新张量的数据可能会影响原始张量。
- 内存连续性：如果原始张量的内存是连续的，reshape 通常会返回一个视图。如果不是连续的，reshape 可能会返回一个新的张量。
- 使用 -1 来自动推断某个维度的大小。例如，如果你想将一个张量重塑为某个形状，但不确定其中一个维度的大小，可以使用 -1：

示例代码

```py
import torch

# 创建一个一维张量
original_tensor = torch.arange(12)  # 生成一个包含 0 到 11 的一维张量
print("原始张量:", original_tensor)

# 重塑为 3x4 的二维张量
reshaped_tensor = original_tensor.reshape(3, 4)
print("重塑后的张量:\n", reshaped_tensor)

reshaped_tensor = original_tensor.reshape(3, -1)  # 自动计算第二维的大小
```

== 2 广播语义

如果 PyTorch 操作支持广播，则可以自动扩展其张量参数，使其大小相等（而无需复制数据）。

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

