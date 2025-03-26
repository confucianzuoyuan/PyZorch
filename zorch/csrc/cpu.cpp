#include "cpu.h"
#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void add_tensor_cpu(Tensor *tensor1, Tensor *tensor2, float *result_data) {
  for (int i = 0; i < tensor1->size; i++) {
    result_data[i] = tensor1->data[i] + tensor2->data[i];
  }
}

/// 广播操作
/// >>> import torch
/// >>> torch.tensor([1,2,3]) + 1
/// tensor([2, 3, 4])
///
/// # 创建一个 2x3 的张量
/// tensor_a = torch.tensor([[1, 2, 3],
///                           [4, 5, 6]])
/// 
/// # 创建一个标量
/// scalar_b = 10
/// tensor([[11, 12, 13],
///         [14, 15, 16]])
///
/// import torch
/// 
/// # 创建一个 2x3 的张量, ndim = 2
/// tensor_a = torch.tensor([[1, 2, 3],
///                           [4, 5, 6]])
/// 
/// # 创建一个 1x3 的张量, ndim = 1
/// tensor_b = torch.tensor([10, 20, 30])
/// 
/// # 广播加法, tensor_b 的形状是 (1, 3) 广播到 (2, 3)
/// # 逐点相加
/// result = tensor_a + tensor_b
/// 
/// print(result)
/// tensor([[11, 22, 33],
///         [14, 25, 36]])
///
/// # ta: 2x3
/// >>> ta = torch.tensor([[1,2,3],[4,5,6]])
/// # tb: 2x1x3
/// >>> tb = torch.tensor([[[10,20,30]],[[40,50,60]]])
/// # ta + tb: 2x2x3
/// >>> ta + tb
/// tensor([[[11, 22, 33],
///          [14, 25, 36]],
/// 
///         [[41, 52, 63],
///          [44, 55, 66]]])
/// 计算过程：
/// max_ndim = 3
/// broadcasted_shape = [0, 0, 0]
/// i = 0:
///   dim1 = 3
///   dim2 = 3
///   broadcasted_shape[2] = 3
/// i = 1:
///   dim1 = 2
///   dim2 = 1
///   broadcasted_shape[1] = 2
/// i = 2:
///   dim1 = 1
///   dim2 = 2
///   broadcasted_shape[0] = 2
/// ===> broadcasted_shape = [2, 2, 3]
/// ===> broadcasted_size  = 2x2x3 = 12
void add_broadcasted_tensor_cpu(Tensor *tensor1, Tensor *tensor2,
                                float *result_data, int *broadcasted_shape,
                                int broadcasted_size) {
  // 取两个张量的最大维度
  // max_ndim = 3
  int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

  // 计算广播运算的步幅
  int *strides1 = (int *)malloc(max_ndim * sizeof(int));
  int *strides2 = (int *)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  // i = 2:
  //   dim1 = 1; dim2 = 3; strides1[2] = 0; strides2[2] = 1; stride1 = 1; stride2 = 3;
  // i = 1:
  //   dim1 = 2; dim2 = 1; strides1[1] = 1; strides2[1] = 0; stride1 = 2; stride2 = 3;
  // i = 0:
  //   dim1 = 3; dim2 = 2; strides1[0] = 0; strides2[0] = 3; stride1 = 2; stride2 = 6;
  // strides1 = [0, 1, 0];
  // strides2 = [3, 0, 1];
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

  // 计算逐点相加
  // i = 6:
  //   j = 2:
  //     pos = 0; linear_index = 2; index1 = 0; index2 = 0;
  //   j = 1:
  //     pos = 0; linear_index = 1; index1 = 0; index2 = 0;
  //   j = 0:
  //     pos = 1; linear_index = 0; index1 = 0; index2 = 3;
  // result_data[6] = 1 + 40 = 41
  for (int i = 0; i < broadcasted_size; i++) {
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
    result_data[i] = tensor1->data[index1] + tensor2->data[index2];
  }

  // 释放内存
  free(strides1);
  free(strides2);
}