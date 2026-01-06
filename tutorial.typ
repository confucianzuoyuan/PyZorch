张量在内存中的表示是一个一维数组。

所以访问高维张量中的元素的公式如下：

$
  T[i][j][k]...[z] = "array"[i times "stride"[0] + j times "stride"[1] + k times "stride"[2] + ... + z times "stride"[n-1]]
$

其中每个维度的步幅可以用下一维张量形状的乘积计算：

$
  "stride"[k] = product_(i=k+1)^(N-1)"shape"[i]
$

其中$"stride"[n-1]=1$。

张量的形状如果是$[5, 4, 8]$，$"strides" = [4 times 8, 8, 1] = [32, 8, 1]$。

