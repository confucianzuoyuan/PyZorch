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

