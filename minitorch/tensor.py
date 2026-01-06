import ctypes
import os


class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_int),
        ("device", ctypes.c_char_p),
    ]


class Tensor:
    # 类级别初始化 C 库
    _lib_initialized = False
    _C = None

    @classmethod
    def _init_lib(cls):
        """初始化 C 库（只执行一次）"""
        if cls._lib_initialized:
            return

        libtensor_dir = os.path.dirname(os.path.abspath(__file__))
        cls._C = ctypes.CDLL(os.path.join(libtensor_dir, "libtensor.so"))

        # 设置 create_tensor 函数签名
        cls._C.create_tensor.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_char_p,
        ]
        cls._C.create_tensor.restype = ctypes.POINTER(CTensor)

        # 设置 get_item 函数签名
        cls._C.get_item.argtypes = [
            ctypes.POINTER(CTensor),
            ctypes.POINTER(ctypes.c_int),
        ]
        cls._C.get_item.restype = ctypes.c_float

        # 设置 free_tensor 函数签名（如果有）
        if hasattr(cls._C, 'free_tensor'):
            cls._C.free_tensor.argtypes = [ctypes.POINTER(CTensor)]
            cls._C.free_tensor.restype = None

        cls._lib_initialized = True

    def __init__(self, data, device="cpu", requires_grad=False):
        """
        创建张量

        Args:
            data: 数值、列表或嵌套列表
            device: 设备类型 ("cpu" 或 "cuda")
            requires_grad: 是否需要梯度
        """
        # 初始化 C 库
        Tensor._init_lib()

        # 解析数据和形状
        flat_data, shape = self._parse_data(data)

        # 保存属性
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.device = device
        self.numel = len(flat_data)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

        # 创建 ctypes 数组（保持引用避免被 GC）
        self._data_buffer = (ctypes.c_float * len(flat_data))(*flat_data)
        self._shape_buffer = (ctypes.c_int * self.ndim)(*
                                                        shape) if self.ndim > 0 else None

        # 调用 C 函数创建张量
        self.tensor = Tensor._C.create_tensor(
            self._data_buffer,
            self._shape_buffer if self._shape_buffer else ctypes.POINTER(
                ctypes.c_int)(),
            ctypes.c_int(self.ndim),
            device.encode("utf-8"),
        )

        if not self.tensor:
            raise RuntimeError("Failed to create tensor in C")

    @staticmethod
    def _parse_data(data):
        """
        解析数据，返回扁平化的数据和形状

        Args:
            data: 标量、列表或嵌套列表

        Returns:
            (flat_data, shape): 扁平化的数据列表和形状列表
        """
        # 处理标量
        if isinstance(data, (int, float)):
            return [float(data)], []

        # 处理列表
        if isinstance(data, list):
            # 检查是否为空列表
            if len(data) == 0:
                raise ValueError("Cannot create tensor from empty list")

            # 递归解析第一个元素以确定内部形状
            if isinstance(data[0], (int, float)):
                # 1D 列表
                return [float(x) for x in data], [len(data)]
            elif isinstance(data[0], list):
                # 多维列表
                flat_data = []
                first_inner_data, inner_shape = Tensor._parse_data(data[0])
                flat_data.extend(first_inner_data)

                # 验证所有元素形状一致
                expected_len = len(first_inner_data)
                for i, item in enumerate(data[1:], 1):
                    inner_data, item_shape = Tensor._parse_data(item)
                    if item_shape != inner_shape:
                        raise ValueError(
                            f"Inconsistent shape at index {i}: "
                            f"expected {inner_shape}, got {item_shape}"
                        )
                    if len(inner_data) != expected_len:
                        raise ValueError(
                            f"Inconsistent data length at index {i}: "
                            f"expected {expected_len}, got {len(inner_data)}"
                        )
                    flat_data.extend(inner_data)

                return flat_data, [len(data)] + inner_shape
            else:
                raise TypeError(
                    f"Unsupported data type in list: {type(data[0])}")

        raise TypeError(f"Unsupported data type: {type(data)}")

    def __getitem__(self, indices):
        """获取张量元素"""
        # 处理标量张量
        if self.ndim == 0:
            if indices != () and indices != slice(None):
                raise IndexError("Invalid index for scalar tensor")
            c_indices = (ctypes.c_int * 0)()
            return Tensor._C.get_item(self.tensor, c_indices)

        # 标准化索引
        if not isinstance(indices, tuple):
            indices = (indices,)

        # 过滤掉省略号和切片（简化版本，只支持整数索引）
        int_indices = []
        for idx in indices:
            if isinstance(idx, int):
                int_indices.append(idx)
            elif isinstance(idx, slice):
                raise NotImplementedError("Slice indexing not yet supported")
            else:
                raise TypeError(f"Unsupported index type: {type(idx)}")

        # 检查索引维度
        if len(int_indices) > self.ndim:
            raise IndexError(
                f"Too many indices: tensor has {self.ndim} dimensions, "
                f"got {len(int_indices)}"
            )

        # 处理负索引
        normalized_indices = []
        for i, idx in enumerate(int_indices):
            if idx < 0:
                idx = self.shape[i] + idx
            if idx < 0 or idx >= self.shape[i]:
                raise IndexError(
                    f"Index {idx} out of bounds for dimension {i} "
                    f"with size {self.shape[i]}"
                )
            normalized_indices.append(idx)

        # 转换为 C 数组
        c_indices = (ctypes.c_int * len(normalized_indices)
                     )(*normalized_indices)
        value = Tensor._C.get_item(self.tensor, c_indices)

        # 如果索引数量少于维度，应该返回子张量（这里简化为只返回标量）
        if len(normalized_indices) < self.ndim:
            raise NotImplementedError("Partial indexing not yet supported")

        return value

    def __str__(self):
        """字符串表示"""
        if self.ndim == 0:
            return f"tensor({self[()]:.4f})"

        # 递归构建多维数组的字符串表示
        def format_array(indices, depth=0):
            if depth == self.ndim:
                return f"{self[indices]:.4f}"

            dim_size = self.shape[depth]

            # 控制显示长度
            if dim_size > 6:
                items = []
                for i in range(3):
                    items.append(format_array(indices + (i,), depth + 1))
                items.append('...')
                for i in range(dim_size - 3, dim_size):
                    items.append(format_array(indices + (i,), depth + 1))
            else:
                items = [format_array(indices + (i,), depth + 1)
                         for i in range(dim_size)]

            # 格式化
            if depth == self.ndim - 1:
                # 最内层：单行显示
                return '[' + ', '.join(items) + ']'
            else:
                # 外层：多行显示
                indent = ' ' * (depth + 1)
                sep = ',\n' + indent
                return '[' + sep.join(items) + ']'

        tensor_str = format_array(())

        # 添加元数据
        parts = [f"tensor({tensor_str}"]
        if self.device != "cpu":
            parts.append(f"device='{self.device}'")
        if self.requires_grad:
            parts.append(f"requires_grad=True")

        return ', '.join(parts) + ')'

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        """析构函数：释放 C 端内存"""
        if hasattr(self, 'tensor') and self.tensor and hasattr(Tensor._C, 'free_tensor'):
            Tensor._C.free_tensor(self.tensor)

    def tolist(self):
        """转换为 Python 列表"""
        if self.ndim == 0:
            return self[()]

        def build_list(indices, depth=0):
            if depth == self.ndim:
                return self[indices]

            return [build_list(indices + (i,), depth + 1)
                    for i in range(self.shape[depth])]

        return build_list(())

    def item(self):
        """获取标量值（仅用于标量张量或单元素张量）"""
        if self.numel != 1:
            raise ValueError(
                f"Only one element tensors can be converted to Python scalars, "
                f"got {self.numel} elements"
            )

        if self.ndim == 0:
            return self[()]
        else:
            # 单元素张量
            return self[(0,) * self.ndim]


def tensor(data, device="cpu", requires_grad=False):
    """
    创建张量的工厂函数

    Args:
        data: 数值、列表或嵌套列表
        device: 设备类型
        requires_grad: 是否需要梯度

    Returns:
        Tensor 对象

    Examples:
        >>> t1 = tensor(3.14)
        >>> t2 = tensor([1, 2, 3])
        >>> t3 = tensor([[1, 2], [3, 4]], requires_grad=True)
    """
    return Tensor(data=data, device=device, requires_grad=requires_grad)


def ones_like(tensor, device=None, requires_grad=False):
    """
    创建一个与给定张量形状相同、元素全为 1 的新张量

    Args:
        tensor: 参考张量
        device: 设备类型，如果为 None 则使用参考张量的设备
        requires_grad: 是否需要梯度

    Returns:
        新的张量，形状与 tensor 相同，所有元素为 1.0

    Examples:
        >>> t = tensor([[1, 2], [3, 4]])
        >>> ones_like(t)
        tensor([[1.0000, 1.0000],
                [1.0000, 1.0000]])
    """
    if device is None:
        device = tensor.device

    # 处理标量张量
    if tensor.ndim == 0:
        return Tensor(1.0, device=device, requires_grad=requires_grad)

    # 递归创建嵌套列表
    def create_ones(shape):
        if len(shape) == 0:
            return 1.0
        if len(shape) == 1:
            return [1.0] * shape[0]
        return [create_ones(shape[1:]) for _ in range(shape[0])]

    data = create_ones(tensor.shape)
    return Tensor(data, device=device, requires_grad=requires_grad)


def zeros_like(tensor, device=None, requires_grad=False):
    """
    创建一个与给定张量形状相同、元素全为 0 的新张量

    Args:
        tensor: 参考张量
        device: 设备类型，如果为 None 则使用参考张量的设备
        requires_grad: 是否需要梯度

    Returns:
        新的张量，形状与 tensor 相同，所有元素为 0.0

    Examples:
        >>> t = tensor([[1, 2], [3, 4]])
        >>> zeros_like(t)
        tensor([[0.0000, 0.0000],
                [0.0000, 0.0000]])
    """
    if device is None:
        device = tensor.device

    if tensor.ndim == 0:
        return Tensor(0.0, device=device, requires_grad=requires_grad)

    def create_zeros(shape):
        if len(shape) == 0:
            return 0.0
        if len(shape) == 1:
            return [0.0] * shape[0]
        return [create_zeros(shape[1:]) for _ in range(shape[0])]

    data = create_zeros(tensor.shape)
    return Tensor(data, device=device, requires_grad=requires_grad)


def full_like(tensor, fill_value, device=None, requires_grad=False):
    """
    创建一个与给定张量形状相同、元素全为指定值的新张量

    Args:
        tensor: 参考张量
        fill_value: 填充值
        device: 设备类型，如果为 None 则使用参考张量的设备
        requires_grad: 是否需要梯度

    Returns:
        新的张量，形状与 tensor 相同，所有元素为 fill_value

    Examples:
        >>> t = tensor([[1, 2], [3, 4]])
        >>> full_like(t, 3.14)
        tensor([[3.1400, 3.1400],
                [3.1400, 3.1400]])
    """
    if device is None:
        device = tensor.device

    fill_value = float(fill_value)

    if tensor.ndim == 0:
        return Tensor(fill_value, device=device, requires_grad=requires_grad)

    def create_filled(shape):
        if len(shape) == 0:
            return fill_value
        if len(shape) == 1:
            return [fill_value] * shape[0]
        return [create_filled(shape[1:]) for _ in range(shape[0])]

    data = create_filled(tensor.shape)
    return Tensor(data, device=device, requires_grad=requires_grad)


def ones(shape, device="cpu", requires_grad=False):
    """
    创建指定形状、元素全为 1 的张量

    Args:
        shape: 形状，可以是整数或整数元组
        device: 设备类型
        requires_grad: 是否需要梯度

    Returns:
        新的张量，所有元素为 1.0

    Examples:
        >>> ones(3)
        tensor([1.0000, 1.0000, 1.0000])
        >>> ones((2, 3))
        tensor([[1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000]])
    """
    # 标准化形状
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)

    # 处理标量（空形状）
    if len(shape) == 0:
        return Tensor(1.0, device=device, requires_grad=requires_grad)

    def create_ones(s):
        if len(s) == 0:
            return 1.0
        if len(s) == 1:
            return [1.0] * s[0]
        return [create_ones(s[1:]) for _ in range(s[0])]

    data = create_ones(shape)
    return Tensor(data, device=device, requires_grad=requires_grad)


def zeros(shape, device="cpu", requires_grad=False):
    """
    创建指定形状、元素全为 0 的张量

    Args:
        shape: 形状，可以是整数或整数元组
        device: 设备类型
        requires_grad: 是否需要梯度

    Returns:
        新的张量，所有元素为 0.0

    Examples:
        >>> zeros(3)
        tensor([0.0000, 0.0000, 0.0000])
        >>> zeros((2, 3))
        tensor([[0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000]])
    """
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)

    if len(shape) == 0:
        return Tensor(0.0, device=device, requires_grad=requires_grad)

    def create_zeros(s):
        if len(s) == 0:
            return 0.0
        if len(s) == 1:
            return [0.0] * s[0]
        return [create_zeros(s[1:]) for _ in range(s[0])]

    data = create_zeros(shape)
    return Tensor(data, device=device, requires_grad=requires_grad)


def full(shape, fill_value, device="cpu", requires_grad=False):
    """
    创建指定形状、元素全为指定值的张量

    Args:
        shape: 形状，可以是整数或整数元组
        fill_value: 填充值
        device: 设备类型
        requires_grad: 是否需要梯度

    Returns:
        新的张量，所有元素为 fill_value

    Examples:
        >>> full(3, 3.14)
        tensor([3.1400, 3.1400, 3.1400])
        >>> full((2, 3), -1)
        tensor([[-1.0000, -1.0000, -1.0000],
                [-1.0000, -1.0000, -1.0000]])
    """
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)

    fill_value = float(fill_value)

    if len(shape) == 0:
        return Tensor(fill_value, device=device, requires_grad=requires_grad)

    def create_filled(s):
        if len(s) == 0:
            return fill_value
        if len(s) == 1:
            return [fill_value] * s[0]
        return [create_filled(s[1:]) for _ in range(s[0])]

    data = create_filled(shape)
    return Tensor(data, device=device, requires_grad=requires_grad)


def empty(shape, device="cpu", requires_grad=False):
    """
    创建指定形状的未初始化张量（实际上初始化为 0）

    Args:
        shape: 形状，可以是整数或整数元组
        device: 设备类型
        requires_grad: 是否需要梯度

    Returns:
        新的张量

    Examples:
        >>> empty(3)
        tensor([0.0000, 0.0000, 0.0000])
    """
    # 在 Python 实现中，empty 和 zeros 行为相同
    return zeros(shape, device=device, requires_grad=requires_grad)


def empty_like(tensor, device=None, requires_grad=False):
    """
    创建一个与给定张量形状相同的未初始化张量

    Args:
        tensor: 参考张量
        device: 设备类型，如果为 None 则使用参考张量的设备
        requires_grad: 是否需要梯度

    Returns:
        新的张量

    Examples:
        >>> t = tensor([[1, 2], [3, 4]])
        >>> empty_like(t)
        tensor([[0.0000, 0.0000],
                [0.0000, 0.0000]])
    """
    return zeros_like(tensor, device=device, requires_grad=requires_grad)
