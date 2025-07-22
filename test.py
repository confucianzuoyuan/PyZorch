import zorch
from zorch.autograd.functions import Add, Exp, Square, add, exp, square

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

# A = Square()
# B = Exp()
# C = Square()

# x = zorch.Tensor(0.5)
# a = A(x)
# b = B(a)
# y = C(b)

# print("y.creator: ", y)

# y.grad = zorch.Tensor(1.0)
# y.backward()
# print(x.grad)

# x = zorch.Tensor(0.5)
# y = square(exp(square(x)))
# y.backward()
# print(x.grad)

x = zorch.Tensor(2.0)
y = zorch.Tensor(3.0)
z = add(square(x), square(y))
z.backward()
print(z)
print(x.grad)
print(y.grad)

x = zorch.Tensor(3.0)
y = add(x, x)
print('y', y)

y.backward()
print('x.grad', x.grad)

x = zorch.Tensor(3.0)
y = add(x, x)
y.backward()
print(x.grad)  # 2.0
# 第 2 个计算(使用同一个 x 进行不同的计算)
x.cleargrad()
y = add(add(x, x), x)
y. backward()
print(x.grad)  # 3.0

x = zorch.Tensor(2.0)
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y)
print(x.grad)