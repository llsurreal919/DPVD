import sys

sys.path.append('../')
from nets.PlotNeuralNet.pycore.tikzeng import *
from nets.PlotNeuralNet.pycore.blocks import *




arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # input

    # Encoder
    to_ConvConvRelu(name='conv1', s_filer=500, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)", width=(2, 2),
                    height=40, depth=40),
    to_Pool(name="pool1", offset="(0,0,0)", to="(conv1-east)", width=1, height=32, depth=32, opacity=0.5),

    *block_2ConvPool(name='b2', bottom='pool1', top='pool_b2', s_filer=256, n_filer=128, offset="(1,0,0)",
                     size=(32, 32, 3.5), opacity=0.5),
    *block_2ConvPool(name='b3', bottom='pool_b2', top='pool_b3', s_filer=128, n_filer=256, offset="(1,0,0)",
                     size=(25, 25, 4.5), opacity=0.5),
    *block_2ConvPool(name='b4', bottom='pool_b3', top='pool_b4', s_filer=64, n_filer=512, offset="(1,0,0)",
                     size=(16, 16, 5.5), opacity=0.5),

    # Bottleneck
    to_ConvConvRelu(name='bottleneck', s_filer=32, n_filer=(1024, 1024), offset="(2,0,0)", to="(pool_b4-east)",
                    width=(8, 8), height=8, depth=8, caption="Bottleneck"),
    to_connection("pool_b4", "bottleneck"),

    # Decoder
    *block_Unconv(name="b6", bottom="bottleneck", top='end_b6', s_filer=64, n_filer=512, offset="(2.1,0,0)",
                  size=(16, 16, 5.0), opacity=0.5),
    to_skip(of='b4', to='ccr_res_b6', pos=1.25),
    *to_Attention(name="att5", to_pos="(b6-east)", s_filer=64, n_filer=512, width=(8, 8), height=8, depth=8,
                  opacity_val=0.5),
    to_connection("end_b6", "att5"),

    *block_Unconv(name="b7", bottom="att5", top='end_b7', s_filer=128, n_filer=256, offset="(2.1,0,0)",
                  size=(25, 25, 4.5), opacity=0.5),
    to_skip(of='b3', to='ccr_res_b7', pos=1.25),
    *to_Attention(name="att4", to_pos="(b7-east)", s_filer=128, n_filer=256, width=(8, 8), height=8, depth=8,
                  opacity_val=0.5),
    to_connection("end_b7", "att4"),

    *block_Unconv(name="b8", bottom="att4", top='end_b8', s_filer=256, n_filer=128, offset="(2.1,0,0)",
                  size=(32, 32, 3.5), opacity=0.5),
    to_skip(of='b2', to='ccr_res_b8', pos=1.25),
    *to_Attention(name="att3", to_pos="(b8-east)", s_filer=256, n_filer=128, width=(8, 8), height=8, depth=8,
                  opacity_val=0.5),
    to_connection("end_b8", "att3"),

    *block_Unconv(name="b9", bottom="att3", top='end_b9', s_filer=512, n_filer=64, offset="(2.1,0,0)",
                  size=(40, 40, 2.5), opacity=0.5),
    to_skip(of='b1', to='ccr_res_b9', pos=1.25),
    *to_Attention(name="att2", to_pos="(b9-east)", s_filer=512, n_filer=64, width=(8, 8), height=8, depth=8,
                  opacity_val=0.5),
    to_connection("end_b9", "att2"),

    to_ConvSoftMax(name="soft1", s_filer=512, offset="(0.75,0,0)", to="(att2-east)", width=1, height=40, depth=40,
                   caption="SOFT"),
    to_connection("att2", "soft1"),

    to_end()
]


def to_input(name, to, s_filer, n_filer, width, height, depth, opacity):
    return r"""
    \pic[shift={%s}] at (%s) {RightBandedBox={
        name=%s,
        caption=%s,
        xlabel={{%d, }},
        zlabel=%d,
        fill=\ConvColor,
        height=%d,
        width=%d,
        depth=%d
        }
    };
    """ % (to, name, name, s_filer, n_filer, width, height, depth, opacity)


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
