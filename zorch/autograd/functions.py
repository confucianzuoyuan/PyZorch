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
        return gradient
