import ctypes
import os


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

            data, shape = self.flatten(data)

            self.shape = shape.copy()

            self._data_ctype = (ctypes.c_float * len(data))(*data.copy())
            self._shape_ctype = (ctypes.c_int * len(shape))(*shape.copy())
            self._ndim_ctype = ctypes.c_int(len(shape))
            self._device_ctype = device.encode('utf-8')

            self.ndim = len(shape)
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
            self.requires_grad = None
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

    def __getitem__(self, indices):
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
        def print_recursively(tensor, depth, index):
            if depth == tensor.ndim - 1:
                result = ""
                for i in range(tensor.shape[-1]):
                    index[-1] = i
                    result += str(tensor[tuple(index)]) + ", "
                return result.strip().strip(",")
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


if __name__ == "__main__":
    print(Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
