"""
计算图可视化示例
"""

import micrograd as mg


def example_simple_graph():
    """示例1：简单计算图"""
    print("=" * 60)
    print("示例1：简单计算图")
    print("=" * 60)
    
    # 创建计算图
    x = mg.tensor([[1, 2]], requires_grad=True)
    x._label = 'x'
    
    y = x * 2
    y._label = 'y = x * 2'
    
    z = y + 1
    z._label = 'z = y + 1'
    
    loss = z.sum()
    loss._label = 'loss'
    
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = {z}")
    print(f"loss = {loss}")
    
    # 反向传播
    loss.backward()
    
    # 可视化
    mg.draw_dot(loss, 'simple_graph', show_shapes=True, show_grad=True)
    mg.draw_simple_dot(loss, 'simple_graph_minimal')
    mg.print_graph_info(loss)
    
    print()


def example_neural_network_graph():
    """示例2：神经网络计算图"""
    print("=" * 60)
    print("示例2：神经网络计算图")
    print("=" * 60)
    
    # 创建数据
    X = mg.tensor([[0, 1], [1, 0]])
    X._label = 'X (input)'
    
    Y = mg.tensor([[1], [1]])
    Y._label = 'Y (target)'
    
    # 创建模型
    model = mg.Sequential(
        mg.Linear(2, 3),
        mg.ReLU(),
        mg.Linear(3, 1),
        mg.Sigmoid()
    )
    
    # 前向传播
    pred = model(X)
    pred._label = 'prediction'
    
    # 计算损失
    criterion = mg.MSELoss()
    loss = criterion(pred, Y)
    loss._label = 'MSE Loss'
    
    print(f"Prediction: {pred}")
    print(f"Loss: {loss}")
    
    # 反向传播
    loss.backward()
    
    # 可视化
    mg.draw_dot(loss, 'neural_network_graph', rankdir='TB')
    mg.print_graph_info(loss)
    
    print()


def example_complex_operations():
    """示例3：复杂操作"""
    print("=" * 60)
    print("示例3：复杂操作计算图")
    print("=" * 60)
    
    # 创建张量
    a = mg.tensor([[1, 2], [3, 4]], requires_grad=True)
    a._label = 'a'
    
    b = mg.tensor([[5, 6], [7, 8]], requires_grad=True)
    b._label = 'b'
    
    # 复杂计算
    c = a + b
    c._label = 'c = a + b'
    
    d = a * b
    d._label = 'd = a * b'
    
    e = c + d
    e._label = 'e = c + d'
    
    f = e.relu()
    f._label = 'f = ReLU(e)'
    
    loss = f.sum()
    loss._label = 'loss'
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"loss = {loss}")
    
    # 反向传播
    loss.backward()
    
    # 可视化
    mg.draw_dot(loss, 'complex_operations', show_shapes=True, show_grad=True)
    mg.print_graph_info(loss)
    
    print()


def example_matrix_multiplication():
    """示例4：矩阵乘法"""
    print("=" * 60)
    print("示例4：矩阵乘法计算图")
    print("=" * 60)
    
    A = mg.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    A._label = 'A (2x3)'
    
    B = mg.tensor([[7, 8], [9, 10], [11, 12]], requires_grad=True)
    B._label = 'B (3x2)'
    
    C = A @ B
    C._label = 'C = A @ B'
    
    loss = C.sum()
    loss._label = 'loss'
    
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = {C}")
    
    loss.backward()
    
    mg.draw_dot(loss, 'matmul_graph', show_shapes=True, show_grad=True)
    mg.print_graph_info(loss)
    
    print()


def example_activation_functions():
    """示例5：激活函数"""
    print("=" * 60)
    print("示例5：激活函数计算图")
    print("=" * 60)
    
    x = mg.tensor([[-1, 0, 1]], requires_grad=True)
    x._label = 'x'
    
    # ReLU
    y1 = x.relu()
    y1._label = 'ReLU(x)'
    
    # Sigmoid
    y2 = x.sigmoid()
    y2._label = 'Sigmoid(x)'
    
    # Tanh
    y3 = x.tanh()
    y3._label = 'Tanh(x)'
    
    # 组合
    y = y1 + y2 + y3
    y._label = 'y = ReLU + Sigmoid + Tanh'
    
    loss = y.sum()
    loss._label = 'loss'
    
    loss.backward()
    
    mg.draw_dot(loss, 'activation_functions', rankdir='TB')
    mg.print_graph_info(loss)
    
    print()


def example_xor_training():
    """示例6：XOR训练过程可视化"""
    print("=" * 60)
    print("示例6：XOR训练过程")
    print("=" * 60)
    
    # 数据
    X = mg.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    X._label = 'X'
    
    Y = mg.tensor([[0], [1], [1], [0]])
    Y._label = 'Y'
    
    # 模型
    model = mg.Sequential(
        mg.Linear(2, 4),
        mg.ReLU(),
        mg.Linear(4, 1),
        mg.Sigmoid()
    )
    
    criterion = mg.MSELoss()
    optimizer = mg.Adam(model.parameters(), lr=0.1)
    
    # 训练几步并可视化
    for epoch in range(3):
        pred = model(X)
        pred._label = f'pred_epoch_{epoch}'
        
        loss = criterion(pred, Y)
        loss._label = f'loss_epoch_{epoch}'
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.data[0]:.6f}")
        
        # 可视化每个epoch的计算图
        mg.draw_simple_dot(loss, f'xor_epoch_{epoch}')
    
    print()


def example_broadcast_graph():
    """示例7：广播机制"""
    print("=" * 60)
    print("示例7：广播机制计算图")
    print("=" * 60)
    
    x = mg.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    x._label = 'x (2x3)'
    
    y = mg.tensor([[10, 20, 30]], requires_grad=True)
    y._label = 'y (1x3)'
    
    z = x + y
    z._label = 'z = x + y (broadcast)'
    
    loss = z.sum()
    loss._label = 'loss'
    
    loss.backward()
    
    mg.draw_dot(loss, 'broadcast_graph', show_shapes=True, show_grad=True)
    mg.print_graph_info(loss)
    
    print()


# 运行所有示例
if __name__ == "__main__":
    print("\n🎨 MicroGrad 计算图可视化示例\n")
    
    example_simple_graph()
    example_neural_network_graph()
    example_complex_operations()
    example_matrix_multiplication()
    example_activation_functions()
    example_xor_training()
    example_broadcast_graph()
    
    print("=" * 60)
    print("✅ 所有可视化示例完成！")
    print("=" * 60)
    print("\n📝 生成的 DOT 文件可以使用以下方式查看：")
    print("   1. 在线查看: https://dreampuf.github.io/GraphvizOnline/")
    print("   2. 安装 Graphviz 后运行: dot -Tpng graph.dot -o graph.png")
    print("   3. 使用 VSCode 插件: Graphviz Preview")
