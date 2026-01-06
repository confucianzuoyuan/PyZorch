"""
计算图可视化模块
生成 Graphviz DOT 格式文件
"""

from .tensor import Tensor


def trace(root):
    """
    从根节点追踪所有节点和边
    
    Args:
        root: 根张量节点
    
    Returns:
        nodes: 所有节点的集合
        edges: 所有边的集合 (from, to)
    """
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges


def draw_dot(root, filename='computation_graph', format='dot', rankdir='LR', 
             show_shapes=True, show_grad=True):
    """
    绘制计算图并保存为 DOT 文件
    
    Args:
        root: 根张量节点
        filename: 输出文件名（不含扩展名）
        format: 输出格式 ('dot', 'png', 'pdf', 'svg' 等)
        rankdir: 图的方向 ('LR'=左到右, 'TB'=上到下)
        show_shapes: 是否显示张量形状
        show_grad: 是否显示梯度信息
    
    Returns:
        dot_string: DOT 格式的字符串
    
    Example:
        >>> x = tensor([[1, 2]], requires_grad=True)
        >>> y = x * 2 + 1
        >>> loss = y.sum()
        >>> loss.backward()
        >>> draw_dot(loss, 'my_graph')
    """
    nodes, edges = trace(root)
    
    # 开始构建 DOT 字符串
    dot_lines = []
    dot_lines.append(f'digraph G {{')
    dot_lines.append(f'    rankdir={rankdir};')
    dot_lines.append(f'    node [shape=record, style=filled];')
    dot_lines.append('')
    
    # 为每个节点分配唯一ID
    node_ids = {node: f'node_{id(node)}' for node in nodes}
    op_ids = {}
    
    # 绘制张量节点
    for node in nodes:
        node_id = node_ids[node]
        
        # 构建标签
        label_parts = []
        
        # 节点名称或标签
        if node._label:
            label_parts.append(f"<b>{node._label}</b>")
        else:
            label_parts.append("tensor")
        
        # 形状信息
        if show_shapes:
            if node.ndim == 0:
                shape_str = "scalar"
            else:
                shape_str = f"shape: {node.shape}"
            label_parts.append(shape_str)
        
        # 数据预览
        if node.size <= 4:
            data_str = f"data: {[round(x, 4) for x in node.data]}"
        else:
            preview = [round(node.data[i], 4) for i in range(min(3, node.size))]
            data_str = f"data: {preview}..."
        label_parts.append(data_str)
        
        # 梯度信息
        if show_grad and node.grad is not None:
            if node.size <= 4:
                grad_str = f"grad: {[round(x, 4) for x in node.grad]}"
            else:
                grad_preview = [round(node.grad[i], 4) for i in range(min(3, node.size))]
                grad_str = f"grad: {grad_preview}..."
            label_parts.append(grad_str)
        
        label = "\\n".join(label_parts)
        
        # 节点颜色
        if node.requires_grad:
            fillcolor = 'lightblue'
        else:
            fillcolor = 'lightgray'
        
        dot_lines.append(f'    {node_id} [label="{label}", fillcolor={fillcolor}];')
    
    dot_lines.append('')
    
    # 绘制操作节点和边
    for n1, n2 in edges:
        # 如果目标节点有操作，创建操作节点
        if n2._op:
            op_id = f'op_{id(n2)}'
            
            # 只创建一次操作节点
            if op_id not in op_ids:
                op_ids[op_id] = True
                
                # 操作节点样式
                op_label = n2._op
                dot_lines.append(
                    f'    {op_id} [label="{op_label}", '
                    f'shape=circle, fillcolor=lightyellow, width=0.8, fixedsize=true];'
                )
            
            # 从输入到操作的边
            dot_lines.append(f'    {node_ids[n1]} -> {op_id};')
            # 从操作到输出的边
            dot_lines.append(f'    {op_id} -> {node_ids[n2]};')
        else:
            # 直接连接（没有操作）
            dot_lines.append(f'    {node_ids[n1]} -> {node_ids[n2]};')
    
    dot_lines.append('}')
    
    # 生成 DOT 字符串
    dot_string = '\n'.join(dot_lines)
    
    # 保存到文件
    dot_filename = f'{filename}.dot'
    with open(dot_filename, 'w', encoding='utf-8') as f:
        f.write(dot_string)
    
    print(f"✅ 计算图已保存到: {dot_filename}")
    
    # 如果安装了 graphviz，尝试渲染
    if format != 'dot':
        try:
            import subprocess
            output_file = f'{filename}.{format}'
            subprocess.run(
                ['dot', f'-T{format}', dot_filename, '-o', output_file],
                check=True,
                capture_output=True
            )
            print(f"✅ 图像已渲染到: {output_file}")
            print(f"   (需要安装 Graphviz: https://graphviz.org/download/)")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"⚠️  无法渲染图像 (需要安装 Graphviz)")
            print(f"   你可以手动运行: dot -T{format} {dot_filename} -o {filename}.{format}")
    
    return dot_string


def draw_simple_dot(root, filename='simple_graph'):
    """
    绘制简化版计算图（只显示操作和形状）
    
    Args:
        root: 根张量节点
        filename: 输出文件名
    
    Returns:
        dot_string: DOT 格式的字符串
    """
    nodes, edges = trace(root)
    
    dot_lines = []
    dot_lines.append('digraph G {')
    dot_lines.append('    rankdir=LR;')
    dot_lines.append('    node [shape=box, style=filled];')
    dot_lines.append('')
    
    node_ids = {node: f'node_{id(node)}' for node in nodes}
    
    # 绘制节点
    for node in nodes:
        node_id = node_ids[node]
        
        # 简化标签
        if node._label:
            label = node._label
        elif node.ndim == 0:
            label = "scalar"
        else:
            label = f"{node.shape}"
        
        fillcolor = 'lightblue' if node.requires_grad else 'lightgray'
        dot_lines.append(f'    {node_id} [label="{label}", fillcolor={fillcolor}];')
    
    dot_lines.append('')
    
    # 绘制边和操作
    for n1, n2 in edges:
        if n2._op:
            op_id = f'op_{id(n2)}'
            dot_lines.append(
                f'    {op_id} [label="{n2._op}", '
                f'shape=circle, fillcolor=lightyellow, width=0.6];'
            )
            dot_lines.append(f'    {node_ids[n1]} -> {op_id} -> {node_ids[n2]};')
        else:
            dot_lines.append(f'    {node_ids[n1]} -> {node_ids[n2]};')
    
    dot_lines.append('}')
    
    dot_string = '\n'.join(dot_lines)
    
    # 保存文件
    dot_filename = f'{filename}.dot'
    with open(dot_filename, 'w', encoding='utf-8') as f:
        f.write(dot_string)
    
    print(f"✅ 简化计算图已保存到: {dot_filename}")
    
    return dot_string


def print_graph_info(root):
    """
    打印计算图的统计信息
    
    Args:
        root: 根张量节点
    """
    nodes, edges = trace(root)
    
    print("=" * 60)
    print("计算图信息")
    print("=" * 60)
    print(f"总节点数: {len(nodes)}")
    print(f"总边数: {len(edges)}")
    
    # 统计操作类型
    ops = {}
    for node in nodes:
        if node._op:
            ops[node._op] = ops.get(node._op, 0) + 1
    
    if ops:
        print(f"\n操作统计:")
        for op, count in sorted(ops.items(), key=lambda x: -x[1]):
            print(f"  {op}: {count}")
    
    # 统计需要梯度的节点
    grad_nodes = sum(1 for node in nodes if node.requires_grad)
    print(f"\n需要梯度的节点: {grad_nodes}/{len(nodes)}")
    
    # 统计已计算梯度的节点
    computed_grad = sum(1 for node in nodes if node.grad is not None)
    print(f"已计算梯度的节点: {computed_grad}/{len(nodes)}")
    
    print("=" * 60)
