import ctypes
import os
from .autograd.functions import *
from typing import Self


class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
        ('device', ctypes.c_char_p)
    ]


class Tensor:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    _C = ctypes.CDLL(os.path.join(module_dir, "libtensor.so"))

    def __init__(self, data=None, device="cpu", requires_grad=False):
        if data != None:
            if isinstance(data, (float, int)):
                data = [data]
                self.shape = ()
                self.ndim = len(self.shape)
                self.device = device

                self._data_ctype = (ctypes.c_float * len(data))(*data.copy())
                self._shape_ctype = ctypes.POINTER(ctypes.c_int)()
                self._ndim_ctype = ctypes.c_int(len(self.shape))
                self._device_ctype = device.encode('utf-8')
            else:
                data, shape = self.flatten(data)

                self.shape = shape.copy()

                self._data_ctype = (ctypes.c_float * len(data))(*data.copy())
                self._shape_ctype = (ctypes.c_int * len(shape))(*shape.copy())
                self._ndim_ctype = ctypes.c_int(len(shape))
                self._device_ctype = device.encode('utf-8')

                self.ndim = len(self.shape)
                self.device = device

                self.numel = 1
                for s in self.shape:
                    self.numel *= s

            self.requires_grad = requires_grad
            self.hooks = []
            self.grad = None
            self.grad_fn = None

            Tensor._C.create_tensor.argtypes = [ctypes.POINTER(
                ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p]
            Tensor._C.create_tensor.restype = ctypes.POINTER(CTensor)

            self.tensor = Tensor._C.create_tensor(
                self._data_ctype,
                self._shape_ctype,
                self._ndim_ctype,
                self._device_ctype
            )
        else:
            self.tensor = None,
            self.shape = None,
            self.ndim = None,
            self.device = device
            self.requires_grad = requires_grad
            self.hooks = []
            self.grad = None
            self.grad_fn = None

    def flatten(self, nested_list):
        def flatten_recursively(nested_list):
            flat_data = []
            shape = []
            if isinstance(nested_list, list):
                for sublist in nested_list:
                    inner_data, inner_shape = flatten_recursively(sublist)
                    flat_data.extend(inner_data)
                shape.append(len(nested_list))
                shape.extend(inner_shape)
            else:
                flat_data.append(nested_list)
            return flat_data, shape

        flat_data, shape = flatten_recursively(nested_list)
        return flat_data, shape

    def __del__(self):
        if hasattr(self, '_data_ctype') and self._data_ctype is not None:
            Tensor._C.delete_strides.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_strides.restype = None
            Tensor._C.delete_strides(self.tensor)

            Tensor._C.delete_device.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_device.restype = None
            Tensor._C.delete_device(self.tensor)

            Tensor._C.delete_tensor.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_tensor.restype = None
            Tensor._C.delete_tensor(self.tensor)
        elif self.tensor is not None:
            Tensor._C.delete_strides.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_strides.restype = None
            Tensor._C.delete_strides(self.tensor)

            Tensor._C.delete_data.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_data.restype = None
            Tensor._C.delete_data(self.tensor)

            Tensor._C.delete_shape.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_shape.restype = None
            Tensor._C.delete_shape(self.tensor)

            Tensor._C.delete_device.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_device.restype = None
            Tensor._C.delete_device(self.tensor)

            Tensor._C.delete_tensor.argtypes = [ctypes.POINTER(CTensor)]
            Tensor._C.delete_tensor.restype = None
            Tensor._C.delete_tensor(self.tensor)

    def __getitem__(self, indices):
        if self.ndim == 0:
            indices = [indices]
            Tensor._C.get_item.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(ctypes.c_int),
            ]
            Tensor._C.get_item.restype = ctypes.c_float

            indices = (ctypes.c_int * len(indices))(*indices)
            value = Tensor._C.get_item(self.tensor, indices)

            return value

        if isinstance(indices, int):
            indices = [indices]
        if len(indices) != self.ndim:
            raise ValueError(
                "Number of indices must match the number of dimensions")

        Tensor._C.get_item.argtypes = [ctypes.POINTER(
            CTensor), ctypes.POINTER(ctypes.c_int)]
        Tensor._C.get_item.restype = ctypes.c_float

        indices = (ctypes.c_int * len(indices))(*indices)
        value = Tensor._C.get_item(self.tensor, indices)

        return value

    def __str__(self):
        if self.ndim == 0:
            return str(self[0])

        def print_recursively(tensor, depth, index):
            if depth == tensor.ndim - 1:
                result = ""
                for i in range(tensor.shape[-1]):
                    index[-1] = i
                    result += str(tensor[tuple(index)]) + ", "
                return result.strip()
            else:
                result = ""
                if depth > 0:
                    result += "\n" + " " * ((depth - 1) * 4)
                for i in range(tensor.shape[depth]):
                    index[depth] = i
                    result += "["
                    result += print_recursively(tensor,
                                                depth + 1, index) + "],"
                    if i < tensor.shape[depth] - 1:
                        result += "\n" + " " * (depth * 4)
                return result.strip(",")

        index = [0] * self.ndim
        result = "tensor(["
        result += print_recursively(self, 0, index)
        result += f"""], device="{self.device}", requires_grad={self.requires_grad})"""
        return result

    def __repr__(self):
        return self.__str__()

    def reshape(self, new_shape):
        # 计算张量中的元素总数量
        total_elements = self.numel

        # 检查 -1 的出现次数，不能大于 1
        if new_shape.count(-1) > 1:
            raise ValueError("Only one dimension can be inferred (set to -1).")

        inferred_dim = None
        known_dims_product = 1
        for dim in new_shape:
            if dim == -1:
                inferred_dim = dim
            else:
                known_dims_product *= dim

        # 如果 -1 在 new_shape 中，则计算 inferred dimension
        # 因为 -1 所在的维度，是需要推断的维度
        # 例如 [2, 3] reshape to [3, -1]
        # 需要推断出 -1 这个维度的大小是 2
        if inferred_dim == -1:
            inferred_dim_size = total_elements // known_dims_product
            new_shape = [inferred_dim_size if dim == -
                         1 else dim for dim in new_shape]

        new_shape_ctype = (ctypes.c_int * len(new_shape))(*new_shape)
        new_ndim_ctype = ctypes.c_int(len(new_shape))

        # `Tensor *reshape_tensor(Tensor *tensor, int *new_shape, int new_ndim)`
        Tensor._C.reshape_tensor.argtypes = [ctypes.POINTER(
            CTensor), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        Tensor._C.reshape_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.reshape_tensor(
            self.tensor, new_shape_ctype, new_ndim_ctype)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = new_shape.copy()
        result_data.ndim = len(new_shape)
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = ReshapeBackward(self)

        return result_data

    def ones_like(self):
        Tensor._C.ones_like_tensor.argtypes = [ctypes.POINTER(CTensor)]
        Tensor._C.ones_like_tensor.restype = ctypes.POINTER(CTensor)
        Tensor._C.ones_like_tensor(self.tensor)

        result_tensor_ptr = Tensor._C.ones_like_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        return result_data

    def zeros_like(self):
        Tensor._C.zeros_like_tensor.argtypes = [ctypes.POINTER(CTensor)]
        Tensor._C.zeros_like_tensor.restype = ctypes.POINTER(CTensor)
        Tensor._C.zeros_like_tensor(self.tensor)

        result_tensor_ptr = Tensor._C.zeros_like_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        return result_data

    def __add__(self, other: Self):
        if isinstance(other, (int, float)):
            other = other * self.ones_like()

        broadcasted_shape_add = []

        # 决定是否需要广播，如果需要广播则获取广播形状
        def broadcast_shape(shape1, shape2):
            if shape1 == shape2:
                return shape1, False

            max_len = max(len(shape1), len(shape2))
            shape1 = [1] * (max_len - len(shape1)) + shape1
            shape2 = [1] * (max_len - len(shape2)) + shape2

            for dim1, dim2 in zip(shape1, shape2):
                if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                    raise ValueError(
                        "Shapes are not compatible for broadcasting")
                broadcasted_shape_add.append(max(dim1, dim2))

            return broadcasted_shape_add, True

        broadcasted_shape_add, needs_broadcasting = broadcast_shape(
            self.shape, other.shape)

        if needs_broadcasting:
            # 如果需要广播，则调用add_broadcasted_tensor函数
            if other.ndim == self.ndim - 1:
                other = other.reshape([1] + other.shape)
            elif self.ndim == other.ndim - 1:
                self = self.reshape([1] + self.shape)

            Tensor._C.add_broadcasted_tensor.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            Tensor._C.add_broadcasted_tensor.restype = ctypes.POINTER(CTensor)

            result_tensor_ptr = Tensor._C.add_broadcasted_tensor(
                self.tensor, other.tensor)

            result_data = Tensor()
            result_data.tensor = result_tensor_ptr
            result_data.shape = broadcasted_shape_add.copy()
            result_data.ndim = len(broadcasted_shape_add)
            result_data.device = self.device
            result_data.numel = 1
            for s in result_data.shape:
                result_data.numel *= s

            result_data.requires_grad = self.requires_grad or other.requires_grad
            if result_data.requires_grad:
                result_data.grad_fn = AddBroadcastedBackward(self, other)
        else:
            # 如果两个张量形状相同，则调用add_tensor
            Tensor._C.add_tensor.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            Tensor._C.add_tensor.restype = ctypes.POINTER(CTensor)

            result_tensor_ptr = Tensor._C.add_tensor(self.tensor, other.tensor)

            result_data = Tensor()
            result_data.tensor = result_tensor_ptr
            result_data.shape = self.shape.copy()
            result_data.ndim = self.ndim

            result_data.device = self.device
            # 在广播的情况下，如果想要计算正确的元素数量
            # 则需要更新numel
            result_data.numel = self.numel

            result_data.requires_grad = self.requires_grad or other.requires_grad
            if result_data.requires_grad:
                result_data.grad_fn = AddBackward(self, other)

        return result_data

    def to(self, device):
        device = str(device)

        self.device = device
        self.device_ctype = self.device.encode('utf-8')

        Tensor._C.to_device.argtypes = [
            ctypes.POINTER(CTensor), ctypes.c_char_p]
        Tensor._C.to_device.restype = None
        Tensor._C.to_device(self.tensor, self.device_ctype)

        return self

    def sum(self, axis=None, keepdim=False) -> Self:
        if axis is not None and axis < 0:
            axis = self.ndim + axis

        if axis == None:
            axis = -1

        if axis > self.ndim - 1:
            raise ValueError(
                f"Error: axis argument {axis} cannot be higher than tensor dimension {self.ndim}")

        Tensor._C.sum_tensor.argtypes = [
            ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_bool]
        Tensor._C.sum_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.sum_tensor(self.tensor, axis, keepdim)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr

        if axis == -1:
            if keepdim:
                result_data.ndim = self.ndim
                result_data.shape = [1] * self.ndim
            else:
                result_data.shape = [1]
                result_data.ndim = 1
        else:
            if keepdim:
                result_data.shape = self.shape[:axis] + \
                    [1] + self.shape[axis+1:]
            else:
                result_data.shape = self.shape[:axis] + self.shape[axis+1:]
            result_data.ndim = len(result_data.shape)

        result_data.device = self.device
        result_data.numel = 1

        for s in result_data.shape:
            result_data.numel *= s

        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = SumBackward(self, axis, keepdim=keepdim)

        return result_data

    def transpose(self, axis1, axis2):
        if axis1 < 0:
            axis1 = self.ndim + axis1
        if axis2 < 0:
            axis2 = self.ndim + axis2

        Tensor._C.transpose_axes_tensor.argtypes = [
            ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int]
        Tensor._C.transpose_axes_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.transpose_axes_tensor(
            self.tensor, axis1, axis2)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()
        result_data.shape[axis1] = self.shape[axis2]
        result_data.shape[axis2] = self.shape[axis1]
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = TransposeBackward(self, axis1, axis2)

        return result_data

    @property
    def T(self):
        Tensor._C.transpose_tensor.argtypes = [ctypes.POINTER(CTensor)]
        Tensor._C.transpose_tensor.restype = ctypes.POINTER(CTensor)

        result_tensor_ptr = Tensor._C.transpose_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = result_tensor_ptr
        result_data.shape = self.shape.copy()[::-1]
        result_data.ndim = self.ndim
        result_data.device = self.device
        result_data.numel = self.numel

        result_data.requires_grad = self.requires_grad
        if result_data.requires_grad:
            result_data.grad_fn = TBackward(self)

        return result_data

    def detach(self):
        self.grad = None
        self.grad_fn = None

        return self

    def backward(self, gradient: Self | None = None):
        # 如果张量不需要梯度，则不进行反向传播
        if not self.requires_grad:
            return

        # 如果不存在要反向传播的梯度
        # 但是张量是 1x1 的矩阵
        # 则将 [1] 作为梯度反向传播回去
        # 这里主要是做错误处理
        if gradient is None:
            if self.shape == [1]:
                gradient = Tensor([1]).to(self.device)
            else:
                raise RuntimeError(
                    "Gradient argument must be specified for non-scalar tensors.")

        stack = [(self, gradient)]
        # 用来存储已经访问过的张量
        visited = set()

        while stack:
            tensor, grad = stack.pop()

            if tensor.grad is None:
                tensor.grad = grad
            else:
                tensor.grad += grad

            if tensor.grad_fn is not None:
                grads = tensor.grad_fn.backward(grad)
                for tensor, grad in zip(tensor.grad_fn.input, grads):
                    if isinstance(tensor, Tensor) and tensor not in visited:
                        stack.append((tensor, grad))
                        visited.add(tensor)
