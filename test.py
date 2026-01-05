from zorch import Variable, Tensor
import zorch.functions as F
from zorch.utils import plot_dot_graph

x = Variable(Tensor(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(x.grad)
