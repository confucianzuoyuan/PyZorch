import zorch
from zorch.functions import add, sin, square
from zorch.variable import Variable


x = Variable(zorch.Tensor(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)

x = Variable(zorch.Tensor([[1, 2, 3], [4, 5, 6]]))
print(x)
print(len(x))

x = Variable(zorch.Tensor(2.0))
y = x + 3.0
print(y)

x = Variable(zorch.Tensor(2.0))
y = 3.0 * x + 1.0
print(y)

x = Variable(zorch.Tensor(2.0))
y = -x
print(y)

x = Variable(zorch.Tensor(2.0))
y = x ** 3
print(y)

x = Variable(zorch.Tensor(2.0))
y1 = 2.0 - x
y2 = x - 1.0
print(y1)
print(y2)


def sphere(x, y):
    z = x**2+y**2
    return z


x = Variable(zorch.Tensor(1.0))
y = Variable(zorch.Tensor(1.0))
z = sphere(x, y)
z.backward()
print(x.grad, y.grad)


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


x = Variable(zorch.Tensor(1.0))
y = Variable(zorch.Tensor(1.0))
z = matyas(x, y)
z.backward()
print(x.grad, y.grad)


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


x = Variable(zorch.Tensor(1.0))
y = Variable(zorch.Tensor(1.0))
z = goldstein(x, y)
z.backward()
print(x.grad, y.grad)

# 数值微分
z = (goldstein(1.000001, 1)-goldstein(1, 1)) / 0.000001
print(z)

x = Variable(zorch.Tensor(3.1415926535897932/4))
y = sin(x)
y.backward()

print(y.data)
print(x.grad)
