from graphviz import Digraph


def draw_unet_with_attention():
    dot = Digraph(comment='UNet with VGG16 Backbone and Attention', format='png')

    # 添加节点
    dot.node('I', 'Input Image (3xHxW)')
    dot.node('F1', 'Feature1 (192xHxW)')
    dot.node('F2', 'Feature2 (384xHxW)')
    dot.node('F3', 'Feature3 (768xHxW)')
    dot.node('F4', 'Feature4 (1024xHxW)')
    dot.node('F5', 'Feature5 (1024xHxW)')

    dot.node('U5', 'UpFeature5 (1024xHxW)')
    dot.node('A5', 'Attention5 (1024xHxW)')
    dot.node('U4', 'UpFeature4 (1024xHxW)')
    dot.node('A4', 'Attention4 (768xHxW)')
    dot.node('U3', 'UpFeature3 (768xHxW)')
    dot.node('A3', 'Attention3 (384xHxW)')
    dot.node('U2', 'UpFeature2 (384xHxW)')
    dot.node('A2', 'Attention2 (192xHxW)')
    dot.node('U1', 'UpFeature1 (192xHxW)')

    dot.node('O', 'Output Image (CxHxW)')

    # 编码器部分
    dot.edge('I', 'F1', label='Conv+ReLU')
    dot.edge('F1', 'F2', label='MaxPool+Conv+ReLU')
    dot.edge('F2', 'F3', label='MaxPool+Conv+ReLU')
    dot.edge('F3', 'F4', label='MaxPool+Conv+ReLU')
    dot.edge('F4', 'F5', label='MaxPool+Conv+ReLU')

    # 解码器部分
    dot.edge('F5', 'U5', label='UpSample+Conv+ReLU')
    dot.edge('U5', 'A5', label='Attention')
    dot.edge('A5', 'U4', label='Pointwise Multiply', style='dashed')
    dot.edge('F4', 'U4', label='Skip Connection', style='dashed')

    dot.edge('U4', 'A4', label='Attention')
    dot.edge('A4', 'U3', label='Pointwise Multiply', style='dashed')
    dot.edge('F3', 'U3', label='Skip Connection', style='dashed')

    dot.edge('U3', 'A3', label='Attention')
    dot.edge('A3', 'U2', label='Pointwise Multiply', style='dashed')
    dot.edge('F2', 'U2', label='Skip Connection', style='dashed')

    dot.edge('U2', 'A2', label='Attention')
    dot.edge('A2', 'U1', label='Pointwise Multiply', style='dashed')
    dot.edge('F1', 'U1', label='Skip Connection', style='dashed')

    dot.edge('U1', 'O', label='Conv2d')

    # 渲染图像
    dot.render('unet_with_attention_structure', view=True)


draw_unet_with_attention()
