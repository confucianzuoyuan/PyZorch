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