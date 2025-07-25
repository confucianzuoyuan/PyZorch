#include "tensor.h"
#include "cpu.h"
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/// cuda.h中使用了类似`__host__`这样的cuda特性，
/// 所以头文件必须放在`<cuda_runtime_api.h>`的后面
#include "cuda.h"

extern "C" {
Tensor *create_tensor(float *data, int *shape, int ndim, char *device) {
  Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
  if (tensor == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  tensor->data = data;
  tensor->shape = shape;
  tensor->ndim = ndim;

  // 设备："cpu" or "cuda"
  // 字符串末尾是 '\0' ，所以需要长度 + 1
  tensor->device = (char *)malloc(strlen(device) + 1);
  if (device != NULL) {
    // 将设备名称拷贝到结构体对应的字段
    strcpy(tensor->device, device);
  } else {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  // 张量在内存中是 1 维数组，数组长度为 shape 中的元素相乘
  tensor->size = 1;
  for (int i = 0; i < ndim; i++) {
    tensor->size *= shape[i];
  }

  // stride: 步幅
  // 是 ndim 大小的数组
  tensor->strides = (int *)malloc(ndim * sizeof(int));
  if (tensor->strides == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  // example 1:
  // shape = [2, 3, 4];
  // ndim = 3;
  // stride = [3 * 4, 4, 1] = [12, 4, 1];
  //
  // example 2:
  // shape = [2, 3];
  // ndim = 2;
  // stride = [3, 1];
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    tensor->strides[i] = stride;
    stride *= shape[i];
  }

  return tensor;
}

/// 按照索引获取张量中的数值
float get_item(Tensor *tensor, int *indices) {
  int index = 0;
  // 计算在数组中真正的索引
  for (int i = 0; i < tensor->ndim; i++) {
    index += indices[i] * tensor->strides[i];
  }

  float result;
  if (strcmp(tensor->device, "cpu") == 0) {
    result = tensor->data[index];
  } else {
    cudaMemcpy(&result, tensor->data + index, sizeof(float),
               cudaMemcpyDeviceToHost);
  }

  return result;
}

/// 删除张量：
/// 1. 判断张量的引用是否为NULL
/// 2. free掉张量
/// 3. 张量的引用置为 NULL
void delete_tensor(Tensor *tensor) {
  if (tensor != NULL) {
    free(tensor);
    tensor = NULL;
  }
}

/// 所有经过动态内存分配的数据结构都需要安全的从内存中删除
void delete_shape(Tensor *tensor) {
  if (tensor->shape != NULL) {
    free(tensor->shape);
    tensor->shape = NULL;
  }
}

/// 将张量中的数据删除，需要判断数据所在位置：cpu还是cuda
void delete_data(Tensor *tensor) {
  if (tensor->data != NULL) {
    if (strcmp(tensor->device, "cpu") == 0) {
      free(tensor->data);
    } else {
      free_cuda(tensor->data);
    }
    tensor->data = NULL;
  }
}

void delete_strides(Tensor *tensor) {
  if (tensor->strides != NULL) {
    free(tensor->strides);
    tensor->strides = NULL;
  }
}

void delete_device(Tensor *tensor) {
  if (tensor->device != NULL) {
    free(tensor->device);
    tensor->device = NULL;
  }
}

void to_device(Tensor *tensor, char *target_device) {
  int device_id = 0;
  char *endptr;

  char *target_device_type;

  // 将字符串转换为long类型的数值
  // "123abc" => 123, endptr指向'a'
  long num = strtol(target_device, &endptr, 10);
  if (*endptr == '\0') {
    device_id = (int)num;
    target_device_type = new char[strlen("cuda") + 1];
    strcpy(target_device_type, "cuda");
  } else {
    target_device_type = new char[strlen("cpu") + 1];
    strcpy(target_device_type, "cpu");
  }

  if ((strcmp(target_device_type, "cuda") == 0) &&
      (strcmp(tensor->device, "cpu") == 0)) {
    cpu_to_cuda(tensor, device_id);
  } else if ((strcmp(target_device_type, "cpu") == 0) &&
             (strcmp(tensor->device, "cuda") == 0)) {
    cuda_to_cpu(tensor);
  }

  free(target_device_type);
}

Tensor *add_tensor(Tensor *tensor1, Tensor *tensor2) {
  if (tensor1->ndim != tensor2->ndim) {
    fprintf(stderr,
            "Tensors must have the same number of dimensions %d and %d for "
            "addition\n",
            tensor1->ndim, tensor2->ndim);
    exit(1);
  }

  if (strcmp(tensor1->device, tensor2->device) != 0) {
    fprintf(stderr, "Tensors must be on the same device: %s and %s\n",
            tensor1->device, tensor2->device);
    exit(1);
  }

  int ndim = tensor1->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    if (tensor1->shape[i] != tensor2->shape[i]) {
      fprintf(stderr,
              "Tensors must have the same shape %d and %d at index %d for "
              "addition\n",
              tensor1->shape[i], tensor2->shape[i], i);
      exit(1);
    }
  }

  if (strcmp(tensor1->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor1->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    add_tensor_cpu(tensor1, tensor2, result_data);
    return create_tensor(result_data, shape, ndim, tensor1->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
    add_tensor_cuda(tensor1, tensor2, result_data);
    return create_tensor(result_data, shape, ndim, tensor1->device);
  }
}

Tensor *sub_tensor(Tensor *tensor1, Tensor *tensor2) {
  if (tensor1->ndim != tensor2->ndim) {
    fprintf(stderr,
            "Tensors must have the same number of dimensions %d and %d for "
            "subtraction\n",
            tensor1->ndim, tensor2->ndim);
    exit(1);
  }

  if (strcmp(tensor1->device, tensor2->device) != 0) {
    fprintf(stderr, "Tensors must be on the same device: %s and %s\n",
            tensor1->device, tensor2->device);
    exit(1);
  }

  int ndim = tensor1->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    if (tensor1->shape[i] != tensor2->shape[i]) {
      fprintf(stderr,
              "Tensors must have the same shape %d and %d at index %d for "
              "subtraction\n",
              tensor1->shape[i], tensor2->shape[i], i);
      exit(1);
    }
    shape[i] = tensor1->shape[i];
  }

  if (strcmp(tensor1->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor1->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    sub_tensor_cpu(tensor1, tensor2, result_data);
    return create_tensor(result_data, shape, ndim, tensor1->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
    sub_tensor_cuda(tensor1, tensor2, result_data);
    return create_tensor(result_data, shape, ndim, tensor1->device);
  }
}

Tensor *reshape_tensor(Tensor *tensor, int *new_shape, int new_ndim) {
  int ndim = new_ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = new_shape[i];
  }

  // 计算 new shape 中所有元素的数量
  int size = 1;
  for (int i = 0; i < new_ndim; i++) {
    size *= shape[i];
  }

  // 检查所有元素的数量是否等于当前张量的大小
  if (size != tensor->size) {
    fprintf(stderr,
            "无法 reshape 张量。新张量的元素数量是：%d，旧张量的元素数量是：%d "
            "。\n",
            size, tensor->size);
    exit(1);
  }

  // 为了实现的简便，reshape会创建1个新的张量
  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    assign_tensor_cpu(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    assign_tensor_cuda(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *zeros_like_tensor(Tensor *tensor) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    zeros_like_tensor_cpu(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    zeros_like_tensor_cuda(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *ones_like_tensor(Tensor *tensor) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    ones_like_tensor_cpu(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    ones_like_tensor_cuda(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

void make_contiguous(Tensor *tensor) {
  int *new_strides = (int *)malloc(tensor->ndim * sizeof(int));
  if (new_strides == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  int stride = 1;
  for (int i = tensor->ndim - 1; i >= 0; i--) {
    new_strides[i] = stride;
    stride *= tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    make_contiguous_tensor_cpu(tensor, result_data, new_strides);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    make_contiguous_tensor_cuda(tensor, result_data, new_strides);
  }
}

Tensor *sum_tensor(Tensor *tensor, int axis, bool keepdim) {
  int ndim;
  int *shape;

  if (axis > tensor->ndim - 1) {
    fprintf(stderr,
            "Error: axis argument %d must be smaller than tensor dimension %d",
            axis, tensor->ndim);
    exit(1);
  }

  if (axis == -1) {
    shape = (int *)malloc(sizeof(int));
    shape[0] = 1;
    ndim = 1;
  } else {
    shape = (int *)malloc((tensor->ndim - 1) * sizeof(int));
    for (int i = 0, j = 0; i < tensor->ndim; ++i) {
      if (i != axis) {
        shape[j++] = tensor->shape[i];
      }
    }
    ndim = tensor->ndim - 1;
  }

  int axis_size = 1;
  for (int i = 0; i < ndim; i++) {
    axis_size *= shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)calloc(axis_size, sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }

    sum_tensor_cpu(tensor, result_data, axis_size, shape, axis);

    if (keepdim) {
      if (axis == -1) {
        ndim = tensor->ndim;
        shape = (int *)malloc((tensor->ndim) * sizeof(int));
        for (int i = 0; i < tensor->ndim; i++) {
          shape[i] = 1;
        }
      } else {
        shape = (int *)malloc((tensor->ndim) * sizeof(int));
        for (int i = 0; i < tensor->ndim; i++) {
          shape[i] = tensor->shape[i];
        }
        shape[axis] = 1;
        ndim = tensor->ndim;
      }
    }

    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    if (axis == -1) {
      cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    } else {
      cudaMalloc((void **)&result_data, axis_size * sizeof(float));
    }

    sum_tensor_cuda(tensor, result_data, axis);

    if (keepdim) {
      if (axis == -1) {
        ndim = tensor->ndim;
        shape = (int *)malloc((tensor->ndim) * sizeof(int));
        for (int i = 0; i < tensor->ndim; i++) {
          shape[i] = 1;
        }
      } else {
        shape = (int *)malloc((tensor->ndim) * sizeof(int));
        for (int i = 0; i < tensor->ndim; i++) {
          shape[i] = tensor->shape[i];
        }
        shape[axis] = 1;
        ndim = tensor->ndim;
      }
    }

    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *transpose_tensor(Tensor *tensor) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[ndim - 1 - i];
  }

  int size = tensor->size;

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    switch (ndim) {
    case 1:
      transpose_1D_tensor_cpu(tensor, result_data);
      break;
    case 2:
      transpose_2D_tensor_cpu(tensor, result_data);
      break;
    case 3:
      transpose_3D_tensor_cpu(tensor, result_data);
      break;
    default:
      fprintf(stderr, "Transpose only supports tensors up to 3 dimensions.\n");
      exit(1);
    }
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, size * sizeof(float));
    switch (ndim) {
    case 1:
      transpose_1D_tensor_cuda(tensor, result_data);
      break;
    case 2:
      transpose_2D_tensor_cuda(tensor, result_data);
      break;
    case 3:
      transpose_3D_tensor_cuda(tensor, result_data);
      break;
    default:
      fprintf(stderr, "Transpose only supports tensors up to 3 dimensions.\n");
      exit(1);
    }
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *transpose_axes_tensor(Tensor *tensor, int axis1, int axis2) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  shape[axis1] = tensor->shape[axis2];
  shape[axis2] = tensor->shape[axis1];

  int size = tensor->size;

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }

    assign_tensor_cpu(tensor, result_data);

    Tensor *new_tensor =
        create_tensor(result_data, shape, ndim, tensor->device);

    for (int i = 0; i < ndim; i++) {
      new_tensor->strides[i] = tensor->strides[i];
    }

    new_tensor->strides[axis1] = tensor->strides[axis2];
    new_tensor->strides[axis2] = tensor->strides[axis1];
    make_contiguous(new_tensor);
    return new_tensor;
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, size * sizeof(float));
    assign_tensor_cuda(tensor, result_data);

    Tensor *new_tensor =
        create_tensor(result_data, shape, ndim, tensor->device);

    for (int i = 0; i < ndim; i++) {
      new_tensor->strides[i] = tensor->strides[i];
    }
    new_tensor->strides[axis1] = tensor->strides[axis2];
    new_tensor->strides[axis2] = tensor->strides[axis1];
    make_contiguous(new_tensor);
    return new_tensor;
  }
}

Tensor *tensor_pow_scalar(Tensor *tensor, float exponent) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    tensor_pow_scalar_cpu(tensor, exponent, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    tensor_pow_scalar_cuda(tensor, exponent, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *sin_tensor(Tensor *tensor) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    sin_tensor_cpu(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    sin_tensor_cuda(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *cos_tensor(Tensor *tensor) {

  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    cos_tensor_cpu(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    cos_tensor_cuda(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *sigmoid_tensor(Tensor *tensor) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    sigmoid_tensor_cpu(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    sigmoid_tensor_cuda(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *log_tensor(Tensor *tensor) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    log_tensor_cpu(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    log_tensor_cuda(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *exp_tensor(Tensor *tensor) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    exp_tensor_cpu(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    exp_tensor_cuda(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *scalar_mul_tensor(Tensor *tensor, float scalar) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    scalar_mul_tensor_cpu(tensor, scalar, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    scalar_mul_tensor_cuda(tensor, scalar, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}

Tensor *elementwise_mul_tensor(Tensor *tensor1, Tensor *tensor2) {
  if (tensor1->ndim != tensor2->ndim) {
    fprintf(stderr,
            "Tensors must have the same number of dimensions %d and %d for "
            "element-wise multiplication\n",
            tensor1->ndim, tensor2->ndim);
    exit(1);
  }

  if (strcmp(tensor1->device, tensor2->device) != 0) {
    fprintf(stderr, "Tensors must be on the same device: %s and %s\n",
            tensor1->device, tensor2->device);
    exit(1);
  }

  int ndim = tensor1->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    if (tensor1->shape[i] != tensor2->shape[i]) {
      fprintf(stderr,
              "Tensors must have the same shape %d and %d at index %d for "
              "element-wise multiplication\n",
              tensor1->shape[i], tensor2->shape[i], i);
      exit(1);
    }
    shape[i] = tensor1->shape[i];
  }

  if (strcmp(tensor1->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor1->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    elementwise_mul_tensor_cpu(tensor1, tensor2, result_data);
    return create_tensor(result_data, shape, ndim, tensor1->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
    elementwise_mul_tensor_cuda(tensor1, tensor2, result_data);
    return create_tensor(result_data, shape, ndim, tensor1->device);
  }
}

Tensor *neg_tensor(Tensor *tensor) {
  int ndim = tensor->ndim;
  int *shape = (int *)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (int i = 0; i < ndim; i++) {
    shape[i] = tensor->shape[i];
  }

  if (strcmp(tensor->device, "cpu") == 0) {
    float *result_data = (float *)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    neg_tensor_cpu(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  } else {
    float *result_data;
    cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
    neg_tensor_cuda(tensor, result_data);
    return create_tensor(result_data, shape, ndim, tensor->device);
  }
}
}