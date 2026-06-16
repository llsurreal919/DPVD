from graphviz import Digraph

# 创建一个有向图
dot = Digraph(comment='The Deep Learning Structure')

# 设置节点形状为矩形，调整节点大小
dot.attr('node', shape='rectangle', width='4', height='3', fixedsize='true')

# 添加节点
dot.node('A', 'Input')
dot.node('B', 'Conv2D 1x1')
dot.node('C', 'GELU')
dot.node('D', 'Conv 5x5\n offsets')
dot.node('E', 'Weighted Summation')
dot.node('F', 'Conv 7x7\n offsets')
dot.node('G', 'Weighted Summation')
dot.node('H', 'Pointwise Multiply')
dot.node('I', 'Conv2D 1x1')
dot.node('J', 'Conv2D 1x1')
dot.node('K', 'Add(Shortcut)')
dot.node('L', 'Shortcut(Clone)')
dot.node('M', 'Output')

# 添加边
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'E')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('E', 'G')
dot.edge('F', 'G')
dot.edge('G', 'H')
dot.edge('C', 'H')
dot.edge('H', 'I')
dot.edge('I', 'J')
dot.edge('J', 'K')
dot.edge('A', 'L')
dot.edge('L', 'K')
dot.edge('K', 'M')

# 渲染图形到文件，并以SVG格式输出
dot.render('structure_graph', view=True, format='png')
