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
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1) # 降维
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.silu = nn.SiLU(inplace=True)
        self.attention = ChannelAttention(out_size)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.silu(outputs)
        outputs = self.conv2(outputs)
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
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

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
        # self.up_conv = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor=2),  # 从 256x256 → 512x512
        #     nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
        #     nn.SiLU(),
        #     nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
        #     nn.SiLU(),
        # )

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

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