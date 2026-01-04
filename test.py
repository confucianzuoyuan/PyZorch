from zorch import Variable, Tensor
from zorch.functions import sin
import math


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


x = Variable(Tensor(0.7853981633974483))
y = my_sin(x)
y.backward()

print(y.data)
print(x.grad)
