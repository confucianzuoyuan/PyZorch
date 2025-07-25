#ifndef CPU_H
#define CPU_H

#include "tensor.h"

/* cpu上的张量相加 */
void add_tensor_cpu(Tensor *tensor1, Tensor *tensor2, float *result_data);

/* cpu上的广播张量相加 */
void add_broadcasted_tensor_cpu(Tensor *tensor1, Tensor *tensor2,
                                float *result_data, int *broadcasted_shape,
                                int broadcasted_size);

/* cpu上的张量相减 */
void sub_tensor_cpu(Tensor *tensor1, Tensor *tensor2, float *result_data);

/* cpu上的广播张量相减 */
void sub_broadcasted_tensor_cpu(Tensor *tensor1, Tensor *tensor2,
                                float *result_data, int *broadcasted_shape,
                                int broadcasted_size);

/* 在 PyTorch 中，创建一个与现有张量具有相同形状和数据类型的全零张量，可以使用
 * torch.zeros_like() 函数。 */
void zeros_like_tensor_cpu(Tensor *tensor, float *result_data);

/* 在 PyTorch 中，创建一个与现有张量具有相同形状和数据类型的全一张量，可以使用
 * torch.ones_like() 函数。 */
void ones_like_tensor_cpu(Tensor *tensor, float *result_data);

void assign_tensor_cpu(Tensor *tensor, float *result_data);

void sum_tensor_cpu(Tensor *tensor, float *result_data, int size,
                    int *result_shape, int axis);

void make_contiguous_tensor_cpu(Tensor *tensor, float *result_data,
                                int *new_strides);

void transpose_1D_tensor_cpu(Tensor *tensor, float *result_data);

void transpose_2D_tensor_cpu(Tensor *tensor, float *result_data);

void transpose_3D_tensor_cpu(Tensor *tensor, float *result_data);
void tensor_pow_scalar_cpu(Tensor *tensor, float exponent, float *result_data);
void cos_tensor_cpu(Tensor *tensor, float *result_data);
void sin_tensor_cpu(Tensor *tensor, float *result_data);
void sigmoid_tensor_cpu(Tensor *tensor, float *result_data);
void log_tensor_cpu(Tensor *tensor, float *result_data);
void exp_tensor_cpu(Tensor *tensor, float *result_data);
void scalar_mul_tensor_cpu(Tensor *tensor, float scalar, float *result_data);
void elementwise_mul_tensor_cpu(Tensor *tensor1, Tensor *tensor2,
                                float *result_data);
void neg_tensor_cpu(Tensor *tensor, float *result_data);
void sub_tensor_cpu(Tensor *tensor1, Tensor *tensor2, float *result_data);

#endif