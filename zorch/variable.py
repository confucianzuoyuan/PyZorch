import zorch


def as_tensor(t) -> "zorch.Tensor":
    if isinstance(t, (float, int)):
        return zorch.Tensor(t)
    return t


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:
    __array_priority__ = 200

    def __init__(self, data: "zorch.Tensor", name=None):
        if data is not None:
            if not isinstance(data, zorch.Tensor):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad: "zorch.Tensor" | None = None
        self.creator: "zorch.Function" = None
        self.generation = 0

    def set_creator(self, func: "zorch.Function"):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = self.data.ones_like()

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            # o 是 weakref,访问需要加()
            gys = [o().grad for o in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y是weakref


Variable.__mul__ = zorch.mul
Variable.__rmul__ = zorch.mul
Variable.__add__ = zorch.add
Variable.__radd__ = zorch.add
Variable.__neg__ = zorch.neg
Variable.__sub__ = zorch.sub
Variable.__rsub__ = zorch.rsub
Variable.__truediv__ = zorch.div
Variable.__rtruediv__ = zorch.rdiv
Variable.__pow__ = pow
