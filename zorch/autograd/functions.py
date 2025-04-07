class ReshapeBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        return [gradient.reshape(self.input[0].shape)]
