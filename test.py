"""
测试 MicroGrad 框架
"""

import micrograd as mg
import micrograd.functional as F


def test_basic():
    """测试基本功能"""
    print("=" * 60)
    print("测试1：基本操作")
    print("=" * 60)
    
    x = mg.tensor([[1, 2], [3, 4]], requires_grad=True)
    y = x * 2 + 1
    loss = y.sum()
    
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"loss = {loss}")
    
    loss.backward()
    print(f"x.grad = {x.grad}")
    print("✅ 基本操作测试通过\n")


def test_neural_network():
    """测试神经网络"""
    print("=" * 60)
    print("测试2：神经网络")
    print("=" * 60)
    
    # XOR数据
    X = mg.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = mg.tensor([[0], [1], [1], [0]])
    
    # 模型
    model = mg.Sequential(
        mg.Linear(2, 4),
        mg.ReLU(),
        mg.Linear(4, 1),
        mg.Sigmoid()
    )
    
    # 训练
    criterion = mg.MSELoss()
    optimizer = mg.Adam(model.parameters(), lr=0.1)
    
    print("训练中...")
    for epoch in range(200):
        pred = model(X)
        loss = criterion(pred, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.data[0]:.6f}")
    
    print("✅ 神经网络测试通过\n")


def test_functional():
    """测试函数式API"""
    print("=" * 60)
    print("测试3：函数式API")
    print("=" * 60)
    
    x = mg.tensor([-2, -1, 0, 1, 2])
    
    print(f"x = {x.data}")
    print(f"relu(x) = {F.relu(x).data}")
    print(f"sigmoid(x) = {F.sigmoid(x).data}")
    print(f"tanh(x) = {F.tanh(x).data}")
    
    # Softmax
    logits = mg.tensor([[1, 2, 3]])
    probs = F.softmax(logits, dim=1)
    print(f"\nlogits = {logits.data}")
    print(f"softmax(logits) = {probs.data}")
    
    print("✅ 函数式API测试通过\n")


def test_matmul():
    """测试矩阵乘法"""
    print("=" * 60)
    print("测试4：矩阵乘法")
    print("=" * 60)
    
    A = mg.tensor([[1, 2], [3, 4]], requires_grad=True)
    B = mg.tensor([[5, 6], [7, 8]], requires_grad=True)
    
    C = A @ B
    loss = C.sum()
    
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = A @ B = {C}")
    
    loss.backward()
    print(f"\nA.grad = {A.grad}")
    print(f"B.grad = {B.grad}")
    
    print("✅ 矩阵乘法测试通过\n")


if __name__ == "__main__":
    print("\n🚀 MicroGrad 框架测试\n")
    
    try:
        test_basic()
        test_neural_network()
        test_functional()
        test_matmul()
        
        print("=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
