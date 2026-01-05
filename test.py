from zorch import Variable, Tensor
import zorch.functions as F
from zorch.utils import plot_dot_graph

x = Variable(Tensor([[1.0, 2.0]]))
print(x.shape)
y = Variable(Tensor([[3.0], [4.0]]))
print(y.shape)

print(x @ y)