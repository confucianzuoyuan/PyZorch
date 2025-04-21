class ReshapeBackward:
    def __init__(self, x):
        self.input = [x]

    # 将传播过来的梯度 gradient 变形成 输入 的形状
    def backward(self, gradient):
        return [gradient.reshape(self.input[0].shape)]


class AddBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    # 加法的反向传播
    def backward(self, gradient):
        return [gradient, gradient]


class AddBroadcastedBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    # 带广播的加法的反向传播
    def backward(self, gradient):
        x, y = self.input
        # 将梯度的形状分别转换成 x 的形状，和 y 的形状
        grad_x = self._reshape_gradient(gradient, x.shape)
        grad_y = self._reshape_gradient(gradient, y.shape)

        return [grad_x, grad_y]

    def _reshape_gradient(self, gradient, shape):
        while len(gradient.shape) > len(shape):
            gradient = gradient.sum(axis=0)

        for i in range(len(shape)):
            if shape[i] == 1:
                gradient = gradient.sum(axis=i, keepdim=True)

        return gradient


class SumBackward:
    def __init__(self, x, axis=None, keepdim=False):
        self.input = [x]
        self.axis = axis
        self.keepdim = keepdim

    def backward(self, gradient):
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

    def backward(self, gradient):
        return [gradient.transpose(self.axis2, self.axis1)]


class TBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        return [gradient.T]
