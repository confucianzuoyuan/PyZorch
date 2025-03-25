#include "tensor.h"
#include "cpu.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

float get_item(Tensor *tensor, int *indices) {
  int index = 0;
  for (int i = 0; i < tensor->ndim; i++) {
    index += indices[i] * tensor->strides[i];
  }

  float result;
  result = tensor->data[index];

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
    free(tensor->data);
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

  float *result_data = (float *)malloc(tensor1->size * sizeof(float));
  if (result_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  add_tensor_cpu(tensor1, tensor2, result_data);
  return create_tensor(result_data, shape, ndim, tensor1->device);
}
}