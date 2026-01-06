#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
Tensor *create_tensor(float *data, int *shape, int ndim, char *device) {
  // 输入验证
  if (data == NULL) {
    fprintf(stderr, "Invalid input parameters\n");
    return NULL;
  }

  for (int i = 0; i < ndim; i++) {
    if (shape[i] <= 0) {
      fprintf(stderr, "Invalid shape: dimension %d is %d\n", i, shape[i]);
      return NULL;
    }
  }

  // 分配张量结构
  Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
  if (tensor == NULL) {
    fprintf(stderr, "Memory allocation failed for tensor\n");
    return NULL;
  }

  // 初始化为 NULL，方便清理
  tensor->data = NULL;
  tensor->shape = NULL;
  tensor->strides = NULL;
  tensor->device = NULL;

  // 计算总大小
  tensor->size = 1;
  for (int i = 0; i < ndim; i++) {
    tensor->size *= shape[i];
  }
  tensor->ndim = ndim;

  // 拷贝数据（深拷贝）
  tensor->data = (float *)malloc(tensor->size * sizeof(float));
  if (tensor->data == NULL) {
    fprintf(stderr, "Memory allocation failed for data\n");
    free_tensor(tensor);
    return NULL;
  }
  memcpy(tensor->data, data, tensor->size * sizeof(float));

  // 拷贝形状
  tensor->shape = (int *)malloc(ndim * sizeof(int));
  if (tensor->shape == NULL) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    free_tensor(tensor);
    return NULL;
  }
  memcpy(tensor->shape, shape, ndim * sizeof(int));

  // 计算步幅
  tensor->strides = (int *)malloc(ndim * sizeof(int));
  if (tensor->strides == NULL) {
    fprintf(stderr, "Memory allocation failed for strides\n");
    free_tensor(tensor);
    return NULL;
  }
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    tensor->strides[i] = stride;
    stride *= shape[i];
  }

  // 拷贝设备信息
  if (device != NULL) {
    tensor->device = (char *)malloc(strlen(device) + 1);
    if (tensor->device == NULL) {
      fprintf(stderr, "Memory allocation failed for device\n");
      free_tensor(tensor);
      return NULL;
    }
    strcpy(tensor->device, device);
  } else {
    // 默认设备
    tensor->device = (char *)malloc(4);
    if (tensor->device == NULL) {
      fprintf(stderr, "Memory allocation failed for device\n");
      free_tensor(tensor);
      return NULL;
    }
    strcpy(tensor->device, "cpu");
  }

  return tensor;
}

// 配套的释放函数
void free_tensor(Tensor *tensor) {
  if (tensor == NULL)
    return;

  if (tensor->data != NULL)
    free(tensor->data);
  if (tensor->shape != NULL)
    free(tensor->shape);
  if (tensor->strides != NULL)
    free(tensor->strides);
  if (tensor->device != NULL)
    free(tensor->device);

  free(tensor);
}

float get_item(Tensor *tensor, int *indices) {
  int index = 0;
  for (int i = 0; i < tensor->ndim; i++) {
    index += indices[i] * tensor->strides[i];
  }

  float result;
  if (strcmp(tensor->device, "cpu") == 0) {
    result = tensor->data[index];
  }

  return result;
}
}