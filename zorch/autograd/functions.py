import zorch
import weakref
import contextlib


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


class ReshapeBackward:
    def __init__(self, x: "zorch.Tensor"):
        self.input = [x]

    # 将传播过来的梯度 gradient 变形成 输入 的形状
    def backward(self, gradient: "zorch.Tensor"):
        return [gradient.reshape(self.input[0].shape)]


class AddBackward:
    def __init__(self, x: "zorch.Tensor", y: "zorch.Tensor"):
        self.input = [x, y]

    # 加法的反向传播
    def backward(self, gradient: "zorch.Tensor"):
        return [gradient, gradient]


class AddBroadcastedBackward:
    def __init__(self, x: "zorch.Tensor", y: "zorch.Tensor"):
        self.input = [x, y]

    # 带广播的加法的反向传播
    def backward(self, gradient: "zorch.Tensor"):
        x, y = self.input
        # 将梯度的形状分别转换成 x 的形状，和 y 的形状
        grad_x = self._reshape_gradient(gradient, x.shape)
        grad_y = self._reshape_gradient(gradient, y.shape)

        return [grad_x, grad_y]

    def _reshape_gradient(self, gradient: "zorch.Tensor", shape):
        while len(gradient.shape) > len(shape):
            gradient = gradient.sum(axis=0)

        for i in range(len(shape)):
            if shape[i] == 1:
                gradient = gradient.sum(axis=i, keepdim=True)

        return gradient


class SumBackward:
    def __init__(self, x: "zorch.Tensor", axis=None, keepdim=False):
        self.input = [x]
        self.axis = axis
        self.keepdim = keepdim

    def backward(self, gradient: "zorch.Tensor"):
        input_shape = self.input[0].shape.copy()
        if self.axis == -1:
            grad_output = float(
                gradient[[0] * len(gradient.shape)]) * self.input[0].ones_like()
        else:
            if self.keepdim:
                input_shape = input_shape[:self.axis] + \
                    [1] + input_shape[self.axis+1:]
            else:
                input_shape = input_shape[:self.axis] + \
                    input_shape[self.axis+1:]

            grad_output_shape = list(input_shape)
            grad_output = gradient.reshape(grad_output_shape)
            grad_output = grad_output + self.input[0].zeros_like()

        return [grad_output]


class TransposeBackward:
    def __init__(self, x, axis1, axis2):
        self.input = [x]
        self.axis1 = axis1
        self.axis2 = axis2

    def backward(self, gradient: "zorch.Tensor"):
        return [gradient.transpose(self.axis2, self.axis1)]


class TBackward:
    def __init__(self, x: "zorch.Tensor"):
        self.input = [x]

    def backward(self, gradient: "zorch.Tensor"):
        return [gradient.T]


class SigmoidBackward:
    def __init__(self, input: "zorch.Tensor"):
        self.input = [input]

    def backward(self, gradient: "zorch.Tensor"):
        sigmoid_x = self.input[0].sigmoid()
        grad_input = gradient * sigmoid_x * (1 - sigmoid_x)

        return [grad_input]


class Function:
    def __call__(self, *inputs):
        inputs = [i for i in inputs]
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        if Config.enable_backprop:
            # 函数的辈分和输出中辈分最大的相同
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)  # 让输出变量保存创造者信息
        self.inputs = inputs  # 保存输入的变量
        self.outputs = [weakref.ref(o) for o in outputs]  # 也保存输出变量
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class LogBackward:
    def __init__(self, x: "zorch.Tensor"):
        self.input = [x]

    def backward(self, gradient: "zorch.Tensor"):
        grad_input = gradient / self.input[0]
        return [grad_input]


class Square(Function):
    def forward(self, x: "zorch.Tensor") -> "zorch.Tensor":
        y = x ** 2
        return y

    def backward(self, gy) -> "zorch.Tensor":
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: "zorch.Tensor") -> "zorch.Tensor":
        y = x.exp()
        return y

    def backward(self, gy) -> "zorch.Tensor":
        x = self.input
        gx = x.exp() * gy
        return gx


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


class Config:
    # 是否启用反向传播
    enable_backprop = True
