"""
MicroGrad 完整使用示例
展示框架的各种功能
"""

import micrograd as mg
import micrograd.functional as F


def example_basic_operations():
    """示例1：基本操作"""
    print("=" * 60)
    print("示例1：基本张量操作")
    print("=" * 60)
    
    # 创建张量
    x = mg.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    print(f"x = {x}")
    print(f"x.shape = {x.shape}")
    print(f"x.ndim = {x.ndim}")
    
    # 算术运算
    y = x * 2 + 1
    print(f"\ny = x * 2 + 1 = {y}")
    
    # 归约操作
    z = y.sum()
    print(f"z = y.sum() = {z}")
    
    # 反向传播
    z.backward()
    print(f"\ndz/dx = {x.grad}")
    print()


def example_activation_functions():
    """示例2：激活函数"""
    print("=" * 60)
    print("示例2：激活函数")
    print("=" * 60)
    
    x = mg.tensor([-2, -1, 0, 1, 2], requires_grad=True)
    
    # ReLU
    y_relu = F.relu(x)
    print(f"ReLU({x.data}) = {y_relu.data}")
    
    # Sigmoid
    y_sigmoid = F.sigmoid(x)
    print(f"Sigmoid({x.data}) = {y_sigmoid.data}")
    
    # Tanh
    y_tanh = F.tanh(x)
    print(f"Tanh({x.data}) = {y_tanh.data}")
    
    # Leaky ReLU
    y_leaky = F.leaky_relu(x, negative_slope=0.1)
    print(f"LeakyReLU({x.data}) = {y_leaky.data}")
    print()


def example_matrix_operations():
    """示例3：矩阵运算"""
    print("=" * 60)
    print("示例3：矩阵运算")
    print("=" * 60)
    
    A = mg.tensor([[1, 2], [3, 4]], requires_grad=True)
    B = mg.tensor([[5, 6], [7, 8]], requires_grad=True)
    
    print(f"A = {A}")
    print(f"B = {B}")
    
    # 矩阵乘法
    C = A @ B
    print(f"\nC = A @ B = {C}")
    
    # 反向传播
    loss = C.sum()
    loss.backward()
    
    print(f"\ndC/dA = {A.grad}")
    print(f"dC/dB = {B.grad}")
    print()


def example_softmax_and_cross_entropy():
    """示例4：Softmax和交叉熵"""
    print("=" * 60)
    print("示例4：Softmax和交叉熵")
    print("=" * 60)
    
    # 模拟3个样本，4个类别的logits
    logits = mg.tensor([
        [2.0, 1.0, 0.1, 0.5],
        [0.5, 2.0, 0.3, 0.2],
        [0.1, 0.2, 2.0, 0.3]
    ], requires_grad=True)
    
    # 目标类别
    targets = mg.tensor([0, 1, 2])
    
    print(f"Logits:\n{logits}")
    print(f"\nTargets: {targets.data}")
    
    # Softmax
    probs = F.softmax(logits, dim=1)
    print(f"\nSoftmax probabilities:")
    for i in range(3):
        print(f"  Sample {i}: {probs.data[i*4:(i+1)*4]}")
    
    # 交叉熵损失
    loss = F.cross_entropy(logits, targets)
    print(f"\nCross Entropy Loss: {loss.data[0]:.4f}")
    
    # 反向传播
    loss.backward()
    print(f"\nGradients shape: {logits.shape}")
    print()


def example_neural_network():
    """示例5：训练神经网络"""
    print("=" * 60)
    print("示例5：训练神经网络（XOR问题）")
    print("=" * 60)
    
    # XOR数据
    X = mg.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    Y = mg.tensor([[0], [1], [1], [0]])
    
    # 构建模型
    model = mg.Sequential(
        mg.Linear(2, 8),
        mg.ReLU(),
        mg.Linear(8, 4),
        mg.ReLU(),
        mg.Linear(4, 1),
        mg.Sigmoid()
    )
    
    # 损失函数和优化器
    criterion = mg.MSELoss()
    optimizer = mg.Adam(model.parameters(), lr=0.1)
    
    # 训练
    print("Training...")
    for epoch in range(500):
        # 前向传播
        pred = model(X)
        loss = criterion(pred, Y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/500, Loss: {loss.data[0]:.6f}")
    
    # 测试
    print("\n测试结果：")
    pred = model(X)
    for i in range(4):
        input_vals = X.data[i*2:(i+1)*2]
        pred_val = pred.data[i]
        target_val = Y.data[i]
        print(f"Input: {input_vals}, Predicted: {pred_val:.4f}, Target: {target_val:.4f}")
    print()


def example_dropout():
    """示例6：Dropout"""
    print("=" * 60)
    print("示例6：Dropout正则化")
    print("=" * 60)
    
    x = mg.tensor([[1, 2, 3, 4, 5]] * 3)
    print(f"Original: {x}")
    
    # 训练模式
    y_train = F.dropout(x, p=0.5, training=True)
    print(f"With dropout (training): {y_train}")
    
    # 推理模式
    y_eval = F.dropout(x, p=0.5, training=False)
    print(f"With dropout (eval): {y_eval}")
    print()


def example_broadcasting():
    """示例7：广播机制"""
    print("=" * 60)
    print("示例7：广播机制")
    print("=" * 60)
    
    # 标量与张量
    x = mg.tensor([[1, 2, 3], [4, 5, 6]])
    y = mg.tensor(10)
    z = x + y
    print(f"x + 10 = {z}")
    
    # 不同形状的张量
    a = mg.tensor([[1, 2, 3]])  # (1, 3)
    b = mg.tensor([[1], [2]])    # (2, 1)
    c = a + b                     # (2, 3)
    print(f"\n[[1, 2, 3]] + [[1], [2]] = {c}")
    print()


def example_reshape():
    """示例8：形状变换"""
    print("=" * 60)
    print("示例8：形状变换")
    print("=" * 60)
    
    x = mg.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    print(f"Original shape: {x.shape}")
    print(f"x = {x}")
    
    # Reshape
    y = x.reshape(4, 2)
    print(f"\nReshaped to (4, 2): {y}")
    
    # Flatten
    z = F.flatten(x)
    print(f"\nFlattened: {z}")
    print()


def example_gradient_accumulation():
    """示例9：梯度累积"""
    print("=" * 60)
    print("示例9：梯度累积")
    print("=" * 60)
    
    x = mg.tensor([1, 2, 3], requires_grad=True)
    
    # 第一次前向和反向
    y1 = (x * 2).sum()
    y1.backward()
    print(f"After first backward: x.grad = {x.grad}")
    
    # 第二次前向和反向（不清零梯度）
    y2 = (x * 3).sum()
    y2.backward()
    print(f"After second backward (accumulated): x.grad = {x.grad}")
    
    # 清零梯度
    x.zero_grad()
    print(f"After zero_grad: x.grad = {x.grad}")
    print()


def example_custom_loss():
    """示例10：自定义损失函数"""
    print("=" * 60)
    print("示例10：自定义损失函数")
    print("=" * 60)
    
    def huber_loss(pred, target, delta=1.0):
        """Huber损失"""
        diff = pred - target
        abs_diff = diff * diff  # 简化：用平方近似绝对值
        
        # 简化版本：只实现L2部分
        return abs_diff.mean()
    
    pred = mg.tensor([1, 2, 3, 4], requires_grad=True)
    target = mg.tensor([1, 2, 5, 4])
    
    loss = huber_loss(pred, target)
    print(f"Predictions: {pred.data}")
    print(f"Targets: {target.data}")
    print(f"Huber Loss: {loss.data[0]:.4f}")
    
    loss.backward()
    print(f"Gradients: {pred.grad}")
    print()


# 运行所有示例
if __name__ == "__main__":
    print("\n🚀 MicroGrad 完整示例集\n")
    
    example_basic_operations()
    example_activation_functions()
    example_matrix_operations()
    example_softmax_and_cross_entropy()
    example_neural_network()
    example_dropout()
    example_broadcasting()
    example_reshape()
    example_gradient_accumulation()
    example_custom_loss()
    
    print("=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)
