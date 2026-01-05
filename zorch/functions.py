import weakref
import contextlib

import zorch


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


class Function:
    def __call__(self, *inputs):
        inputs = [zorch.as_variable(x) for x in inputs]
        # ① 正向传播的计算(主处理)
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [zorch.Variable(zorch.as_tensor(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            # ② 创建连接
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x: "zorch.Tensor") -> "zorch.Tensor":
        raise NotImplementedError()

    def backward(self, gy: "zorch.Tensor") -> "zorch.Tensor":
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs

        return gy * x1, gy * x0


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


class Sin(Function):
    def forward(self, x):
        y = x.sin()
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


class Cos(Function):
    def forward(self, x):
        y = x.cos()
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


class Tanh(Function):
    def forward(self, x):
        y = x.tanh()
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


class Log(Function):
    def forward(self, x):
        y = x.log()
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


class Exp(Function):
    def forward(self, x):
        y = x.exp()
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx


class MatMul(Function):
    def forward(self, x, W):
        y = x @ W
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


def add(x0, x1):
    x1 = zorch.as_tensor(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = zorch.as_tensor(x1)
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = zorch.as_tensor(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = zorch.as_tensor(x1)
    return sub(x1, x0)


def div(x0, x1):
    x1 = zorch.as_tensor(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = zorch.as_tensor(x1)
    return div(x1, x0)


def pow(x, c):
    return Pow(c)(x)


def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def tanh(x):
    return Tanh()(x)


def log(x):
    return Log()(x)


def exp(x):
    return Exp()(x)


def linspace(start, stop, num=50, endpoint=True):
    """
    原生 Python 实现的 linspace 函数
    :param start: 起始值
    :param stop: 终止值
    :param num: 生成的样本数量，默认为 50
    :param endpoint: 是否包含终止值，默认为 True
    """
    if num <= 0:
        return []
    if num == 1:
        return [float(start)]

    # 1. 计算步长 (Step)
    # 如果包含终点，区间被分成 num-1 份
    # 如果不包含终点，区间被分成 num 份
    if endpoint:
        step = (stop - start) / (num - 1)
    else:
        step = (stop - start) / num

    # 2. 生成序列
    # 使用 start + i * step 而不是累加，可以减少浮点数累积误差
    return zorch.Tensor([start + i * step for i in range(num)])
