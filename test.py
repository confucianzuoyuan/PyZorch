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

A = Square()
B = Exp()
C = Square()

x = zorch.Tensor([0.5])
a = A(x)
b = B(a)
y = C(b)

print("y.creator: ", y)

y.grad = zorch.Tensor([1.0])
y.backward()
print(x.grad)
