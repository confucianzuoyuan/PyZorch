import zorch
from zorch.functions import add, square
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
