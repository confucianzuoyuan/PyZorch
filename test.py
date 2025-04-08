import zorch

input = zorch.Tensor(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [10, 11, 12]])
print(input.ones_like())
print(input.zeros_like())
print(input.shape)
print(input.reshape((5, 3)))
