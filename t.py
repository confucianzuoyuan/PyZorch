import micrograd as mg
import micrograd.functional as F

x = mg.tensor(1.0)
print(x.shape)
w = mg.tensor(2.0, requires_grad=True)
b = mg.tensor(3.0, requires_grad=True)

y = F.sigmoid(x * w + b)
print(y.shape)
target = mg.tensor(2.0)

loss = mg.nn.MSELoss()(y, target)
mg.print_graph_info(loss)
loss.backward()

print(w.grad)