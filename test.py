from zorch import Variable, Tensor
import zorch.functions as F
import math
import matplotlib.pyplot as plt

x = Variable(F.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data.numpy(), logs[i].numpy(), label=labels[i])

plt.legend(loc="lower right")
plt.show()
