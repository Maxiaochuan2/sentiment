import torch
import torch.nn as nn
import torch.nn.functional as F
#  cfg 字典包含了不同 VGG 模型的配置，其中 'M' 表示最大池化层，其他数字表示卷积层的输出通道数
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

    #  定义模型Vgg
class VGG(nn.Module):
    def __init__(self, num_classes=10, vgg_name="VGG11"):
        super(VGG, self).__init__()
        #   调用 _make_layers 方法来创建特征提取层（self.features）
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    #  模型的前向传播方法，定义了数据如何通过网络
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

     #   根据 cfg 字典构建网络的卷积层和池化层
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        # 遍历cfg字典中的每个元素。cfg中的每个键值对代表网络中的一层，其中键是层的类型，值是层的特定参数
        for x in cfg:
            #池化层
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                #卷积层
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                #更新下一层的输出通道为当前层的输出通道
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
