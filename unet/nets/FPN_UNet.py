import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class unetUpAttention(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUpAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.silu = nn.SiLU(inplace=True)
        self.attention = ChannelAttention(out_size)

    def forward(self, inputs1, inputs2):
        print("inputs1 size:", inputs1.size())
        print("inputs2 size:", inputs2.size())

        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        print("outputs size after cat:", outputs.size())

        outputs = self.conv1(outputs)
        print("outputs size after conv1:", outputs.size())

        outputs = self.silu(outputs)

        outputs = self.conv2(outputs)
        print("outputs size after conv2:", outputs.size())

        outputs = self.silu(outputs)

        # Apply attention mechanism
        attention = self.attention(outputs)
        outputs = outputs * attention

        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [64, 128, 256, 512, 512]  # 根据 VGG 的通道数设置
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [64, 256, 512, 1024, 2048]  # 根据 ResNet50 的通道数设置
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # FPN Conv layers
        self.fpn_conv1 = nn.Conv2d(in_filters[0], out_filters[0], kernel_size=1)
        self.fpn_conv2 = nn.Conv2d(in_filters[1], out_filters[1], kernel_size=1)
        self.fpn_conv3 = nn.Conv2d(in_filters[2], out_filters[2], kernel_size=1)
        self.fpn_conv4 = nn.Conv2d(in_filters[3], out_filters[3], kernel_size=1)

        # upsampling
        self.up_concat4 = unetUpAttention(in_filters[3], out_filters[3])
        self.up_concat3 = unetUpAttention(in_filters[2], out_filters[2])
        self.up_concat2 = unetUpAttention(in_filters[1], out_filters[1])
        self.up_concat1 = unetUpAttention(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.SiLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # FPN structure
        fpn_feat1 = self.fpn_conv1(feat1)
        fpn_feat2 = self.fpn_conv2(feat2)
        fpn_feat3 = self.fpn_conv3(feat3)
        fpn_feat4 = self.fpn_conv4(feat4)

        up4 = self.up_concat4(feat4, fpn_feat4)
        up3 = self.up_concat3(feat3, fpn_feat3)
        up2 = self.up_concat2(feat2, fpn_feat2)
        up1 = self.up_concat1(feat1, fpn_feat1)

        if self.up_conv is not None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
