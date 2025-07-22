#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
  float *data;
  int *strides;
  int *shape;
  int ndim;
  int size;
  char *device;
} Tensor;

extern "C" {
/* 创建张量 */
Tensor *create_tensor(float *data, int *shape, int ndim, char *device);

/* 删除张量 */
void delete_tensor(Tensor *tensor);

/* 删除步长 */
void delete_strides(Tensor *tensor);

/* 删除形状 */
void delete_shape(Tensor *tensor);

/* 删除设备 */
void delete_device(Tensor *tensor);

/* 使用索引获取张量中的元素 */
float get_item(Tensor *tensor, int *indices);

/* 张量相加运算 */
Tensor *add_tensor(Tensor *tensor1, Tensor *tensor2);

/* 张量相减运算 */
Tensor *sub_tensor(Tensor *tensor1, Tensor *tensor2);

/* 矩阵相乘运算 */
Tensor *matmul_tensor(Tensor *tensor1, Tensor *tensor2);

/* 逐点相乘运算，阿达玛积 */
Tensor *elementwise_mul_tensor(Tensor *tensor1, Tensor *tensor2);

/* 张量和标量的相乘 */
Tensor *scalar_mul_tensor(Tensor *tensor, float scalar);

/* 标量除以张量 */
Tensor *scalar_div_tensor(float scalar, Tensor *tensor);

/* 张量除以标量 */
Tensor *tensor_div_scalar(Tensor *tensor, float scalar);

/* 张量除以张量 */
Tensor *tensor_div_tensor(Tensor *tensor1, Tensor *tensor2);

/* 张量作为底数，标量作为指数 */
Tensor *tensor_pow_scalar(Tensor *tensor, float exponent);

/* 标量作为底数，张量作为指数 */
Tensor *scalar_pow_tensor(float base, Tensor *tensor);

/* 判断两个张量是否相等 */
Tensor *equal_tensor(Tensor *tensor1, Tensor *tensor2);

/* 判断两个广播张量是否相等 */
Tensor *equal_broadcasted_tensor(Tensor *tensor1, Tensor *tensor2);

Tensor *sum_tensor(Tensor *tensor, int axis, bool keepdims);
Tensor *max_tensor(Tensor *tensor, int axis, bool keepdims);
Tensor *min_tensor(Tensor *tensor, int axis, bool keepdims);

Tensor *ones_like_tensor(Tensor *tensor);
Tensor *zeros_like_tensor(Tensor *tensor);
Tensor *sin_tensor(Tensor *tensor);
Tensor *cos_tensor(Tensor *tensor);
Tensor *log_tensor(Tensor *tensor);

/* 改变张量的形状 */
Tensor *reshape_tensor(Tensor *tensor, int *new_shape, int new_ndim);

void to_device(Tensor *tensor, char *device);

/* 在 PyTorch 中，`make_contiguous()`
 * 是一个张量操作，用于确保张量在内存中是连续的。 */
void make_contiguous(Tensor *tensor);

Tensor *transpose_tensor(Tensor *tensor);

Tensor *transpose_axes_tensor(Tensor *tensor, int axis1, int axis2);
Tensor *exp_tensor(Tensor *tensor);
}

#endif