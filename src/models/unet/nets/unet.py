import torch
import torch.nn as nn

from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50
from nets.vgg import VGG16

from pathlib import Path
import sys

for _parent in Path(__file__).resolve().parents:
    direct_plugins = _parent / "plugins"
    modules_plugins = _parent / "modules" / "plugins"
    if direct_plugins.is_dir():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
    if modules_plugins.is_dir():
        modules_root = _parent / "modules"
        if str(modules_root) not in sys.path:
            sys.path.insert(0, str(modules_root))
        break

from plugins import build_attention



class unetUp(nn.Module):
    def __init__(self, in_size, out_size, attention_type=""):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)
        self.attention = build_attention(attention_type, out_size)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = self.attention(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg', attention_type = ""):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        elif backbone == "mobilenetv2":
            self.mobilenet = mobilenetv2(pretrained = pretrained, input_size = 512)
            in_filters = [144, 280, 544, 1376]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50, mobilenetv2.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        bottleneck_channels = {
            "vgg": 512,
            "resnet50": 2048,
            "mobilenetv2": 1280,
        }[backbone]
        self.bottleneck_attention = build_attention(attention_type, bottleneck_channels)

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3], attention_type=attention_type)
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2], attention_type=attention_type)
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1], attention_type=attention_type)
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0], attention_type=attention_type)

        if backbone in {"resnet50", "mobilenetv2"}:
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
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
        elif self.backbone == "mobilenetv2":
            feat1 = self.mobilenet.features[:2](inputs)
            feat2 = self.mobilenet.features[2:4](feat1)
            feat3 = self.mobilenet.features[4:7](feat2)
            feat4 = self.mobilenet.features[7:14](feat3)
            feat5 = self.mobilenet.features[14:](feat4)

        feat5 = self.bottleneck_attention(feat5)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
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
        elif self.backbone == "mobilenetv2":
            for param in self.mobilenet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
        elif self.backbone == "mobilenetv2":
            for param in self.mobilenet.parameters():
                param.requires_grad = True
