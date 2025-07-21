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

__global__ void sum_tensor_cuda_kernel(float *data, float *result_data,
                                       int size);
__global__ void sum_tensor_cuda_kernel_axis(float *data, float *result_data,
                                            int *strides, int *shape, int axis,
                                            int ndim, int axis_stride, int size,
                                            int result_size);
__host__ void sum_tensor_cuda(Tensor *tensor, float *result_data, int axis);

__global__ void make_contiguous_tensor_cuda_kernel(float *data,
                                                   float *result_data, int ndim,
                                                   int size, int *strides,
                                                   int *new_strides);

__host__ void make_contiguous_tensor_cuda(Tensor *tensor, float *result_data,
                                          int *new_strides);

__global__ void transpose_1D_tensor_cuda_kernel(float *data, float *result_data,
                                                int size);
__host__ void transpose_1D_tensor_cuda(Tensor *tensor, float *result_data);

__global__ void transpose_2D_tensor_cuda_kernel(float *data, float *result_data,
                                                int rows, int cols);
__host__ void transpose_2D_tensor_cuda(Tensor *tensor, float *result_data);

__global__ void transpose_3D_tensor_cuda_kernel(float *data, float *result_data,
                                                int batch, int rows, int cols);
__host__ void transpose_3D_tensor_cuda(Tensor *tensor, float *result_data);
__global__ void tensor_pow_scalar_cuda_kernel(float *data, float exponent,
                                              float *result_data, int size);
__host__ void tensor_pow_scalar_cuda(Tensor *tensor, float exponent,
                                     float *result_data);
__global__ void cos_tensor_cuda_kernel(float *data, float *result_data,
                                       int size);
__host__ void cos_tensor_cuda(Tensor *tensor, float *result_data);
#endif