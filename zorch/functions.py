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
    return using_config('enable_backprop', False)


class Function:
    def __call__(self, *inputs) -> "zorch.Variable":
        inputs: list["zorch.Variable"] = [zorch.as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [zorch.Variable(zorch.as_tensor(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x: "zorch.Tensor") -> "zorch.Tensor":
        raise NotImplementedError()

    def backward(self, gy: "zorch.Tensor") -> "zorch.Tensor":
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return x.exp()

    def backward(self, gy):
        x = self.inputs[0].data
        gx = x.exp() * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
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
        x0, x1 = self.inputs[0].data, self.inputs[1].data
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
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


class Sin(Function):
    def forward(self, x):
        y = x.sin()
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * x.cos()
        return gx


def square(x: "zorch.Variable") -> "zorch.Variable":
    return Square()(x)


def exp(x: "zorch.Variable") -> "zorch.Variable":
    return Exp()(x)


def add(x0: "zorch.Variable", x1) -> "zorch.Variable":
    x1: "zorch.Variable" = zorch.as_tensor(x1)
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
