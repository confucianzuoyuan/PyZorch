class ReshapeBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        return [gradient.reshape(self.input[0].shape)]


class AddBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        return [gradient, gradient]


class AddBroadcastedBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        x, y = self.input
        grad_x = self._reshape_gradient(gradient, x.shape)
        grad_y = self._reshape_gradient(gradient, y.shape)

        return [grad_x, grad_y]

    def _reshape_gradient(self, gradient, shape):
        return gradient
