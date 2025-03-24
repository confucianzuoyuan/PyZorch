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
Tensor *create_tensor(float *data, int *shape, int ndim, char *device);
void delete_tensor(Tensor *tensor);
void delete_strides(Tensor *tensor);
void delete_shape(Tensor *tensor);
void delete_device(Tensor *tensor);
float get_item(Tensor *tensor, int *indices);
Tensor *add_tensor(Tensor *tensor1, Tensor *tensor2);
Tensor *sub_tensor(Tensor *tensor1, Tensor *tensor2);
Tensor *matmul_tensor(Tensor *tensor1, Tensor *tensor2);
Tensor *elementwise_mul_tensor(Tensor *tensor1, Tensor *tensor2);
Tensor *equal_tensor(Tensor *tensor1, Tensor *tensor2);

Tensor *sum_tensor(Tensor *tensor, int axis, bool keepdims);
Tensor *max_tensor(Tensor *tensor, int axis, bool keepdims);
Tensor *min_tensor(Tensor *tensor, int axis, bool keepdims);

Tensor *ones_like_tensor(Tensor *tensor);
Tensor *zeros_like_tensor(Tensor *tensor);
Tensor *sin_tensor(Tensor *tensor);
Tensor *cos_tensor(Tensor *tensor);
Tensor *log_tensor(Tensor *tensor);

void to_device(Tensor *tensor, char *device);
}

#endif