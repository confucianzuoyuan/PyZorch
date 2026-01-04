from zorch import Variable, Tensor
from zorch.functions import sin
import math

x = Variable(Tensor(2.0))
print(x / 2)
print(4 / x)

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(Tensor(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)
