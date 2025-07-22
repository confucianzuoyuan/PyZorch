import zorch
from zorch.autograd.functions import Exp, Square

input = zorch.Tensor(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]).to("cuda")
print(input)
print(input.ones_like())
print(input.zeros_like())
print(input.shape)
print(input.reshape([3, 5]))
print(input.sum(1))

a = zorch.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
a.to("cuda")
print(a.shape)
print(a)
print(a.T)

x = zorch.Tensor([1], requires_grad=True)
y = x + x
y.backward()
print('x.grad', x.grad)

x = zorch.Tensor(1)
y = zorch.Tensor(2)
print("x: ", x)
print("y: ", y)
print(x.shape)
print(y.shape)
print(x.ndim)
print(y.ndim)

x = zorch.Tensor([1])
y = x ** 2
print(y)
z = x.cos()
print(z)
z = x.sin()
print(z)

print(x.exp())

A = Square()
B = Exp()
C = Square()

x = zorch.Tensor([0.5])
a = A(x)
b = B(a)
y = C(b)

y.grad = zorch.Tensor([1.0])
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print("x.grad: ", x.grad)
