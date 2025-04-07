#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

__host__ void cpu_to_cuda(Tensor *tensor, int device_id);
__host__ void cuda_to_cpu(Tensor *tensor);
__host__ void free_cuda(float *data);

__global__ void add_tensor_cuda_kernel(float *data1, float *data2,
                                       float *result_data, int size);
__host__ void add_tensor_cuda(Tensor *tensor1, Tensor *tensor2,
                              float *result_data);

__global__ void assign_tensor_cuda_kernel(float *data, float *result_data,
                                          int size);
__host__ void assign_tensor_cuda(Tensor *tensor, float *result_data);

__global__ void zeros_like_tensor_cuda_kernel(float *data, float *result_data,
                                              int size);
__host__ void zeros_like_tensor_cuda(Tensor *tensor, float *result_data);

__global__ void ones_like_tensor_cuda_kernel(float *data, float *result_data,
                                             int size);
__host__ void ones_like_tensor_cuda(Tensor *tensor, float *result_data);

#endif