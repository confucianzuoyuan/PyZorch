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

#endif