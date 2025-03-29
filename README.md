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


