#include "tensor.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/// 每个块的线程数量
#define THREADS_PER_BLOCK 128
/// 瓦片的大小
#define TILE_SIZE 32

__host__ void cpu_to_cuda(Tensor *tensor, int device_id) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (device_id >= deviceCount) {
    fprintf(stderr,
            "Could not send tensor to device %d, only %d devices available\n",
            device_id, deviceCount);
    exit(1);
  }

  cudaSetDevice(device_id);

  float *data_tmp;
  cudaMalloc((void **)&data_tmp, tensor->size * sizeof(float));
  cudaMemcpy(data_tmp, tensor->data, tensor->size * sizeof(float),
             cudaMemcpyHostToDevice);

  tensor->data = data_tmp;

  tensor->device = (char *)malloc(strlen("cuda") + 1);
  strcpy(tensor->device, "cuda");
}

__host__ void cuda_to_cpu(Tensor *tensor) {
  float *data_tmp = (float *)malloc(tensor->size * sizeof(float));

  cudaMemcpy(data_tmp, tensor->data, tensor->size * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(tensor->data);

  tensor->data = data_tmp;

  tensor->device = (char *)malloc(strlen("cpu") + 1);
  strcpy(tensor->device, "cpu");
}

__host__ void free_cuda(float *data) { cudaFree(data); }

__global__ void add_tensor_cuda_kernel(float *data1, float *data2,
                                       float *result_data, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    result_data[i] = data1[i] + data2[i];
  }
}

__host__ void add_tensor_cuda(Tensor *tensor1, Tensor *tensor2,
                              float *result_data) {
  int number_of_blocks =
      (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  add_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(
      tensor1->data, tensor2->data, result_data, tensor1->size);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    exit(1);
  }

  cudaDeviceSynchronize();
}

__global__ void add_broadcasted_tensor_cuda_kernel(float *data1, float *data2,
                                                   float *result_data,
                                                   int *broadcasted_shape,
                                                   int *strides1, int *strides2,
                                                   int max_ndim, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  int index1 = 0, index2 = 0;
  int linear_index = i;
  for (int j = max_ndim - 1; j >= 0; j--) {
    int pos = linear_index % broadcasted_shape[j];
    linear_index /= broadcasted_shape[j];
    if (strides1[j] != 0)
      index1 += pos * strides1[j];
    if (strides2[j] != 0)
      index2 += pos * strides2[j];
  }

  result_data[i] = data1[index1] + data2[index2];
}

__host__ void add_broadcasted_tensor_cuda(Tensor *tensor1, Tensor *tensor2,
                                          float *result_data,
                                          int *broadcasted_shape,
                                          int broadcasted_size) {
  int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

  int *strides1 = (int *)malloc(max_ndim * sizeof(int));
  int *strides2 = (int *)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim - 1; i >= 0; i--) {
    int dim1 =
        i < tensor1->ndim ? tensor1->shape[tensor1->ndim - max_ndim + i] : 1;
    int dim2 =
        i < tensor2->ndim ? tensor2->shape[tensor2->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim2 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim2 == broadcasted_shape[i]) ? dim2 : 1;
  }

  int *d_broadcasted_shape;
  int *d_strides1;
  int *d_strides2;

  cudaMalloc((void **)&d_broadcasted_shape, max_ndim * sizeof(int));
  cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_strides1, max_ndim * sizeof(int));
  cudaMemcpy(d_strides1, strides1, max_ndim * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_strides2, max_ndim * sizeof(int));
  cudaMemcpy(d_strides2, strides2, max_ndim * sizeof(int),
             cudaMemcpyHostToDevice);

  int number_of_blocks =
      (broadcasted_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  add_broadcasted_tensor_cuda_kernel(tensor1->data, tensor2->data, result_data,
                                     d_broadcasted_shape, d_strides1,
                                     d_strides2, max_ndim, broadcasted_size);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    exit(1);
  }

  cudaDeviceSynchronize();
  cudaFree(d_broadcasted_shape);
}