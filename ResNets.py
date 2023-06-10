import torch.nn as nn
from residual_block import ResidualBlock

# Inspired from https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/

class baseResNet(nn.Module):
    def __init__(self):
        super(baseResNet, self).__init__()
        self.block1 = ResidualBlock(3, 12)
        self.pool   = nn.MaxPool2d(kernel_size=2)
        self.block2 = ResidualBlock(12, 32)
        self.fc     = nn.Linear(in_features=64 * 64 * 32, out_features=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = x.view(-1,32*64*64)
        x = self.fc(x)
        return x

    def get_name(self):
        return type(self).__name__

class ResNet(nn.Module):
    def __init__(self, layers, block=ResidualBlock, num_classes = 1):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.dropout2 = nn.Dropout(p=0.25)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.dropout3 = nn.Dropout(p=0.25)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.dropout4 = nn.Dropout(p=0.25)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.layer0(x)
        x = self.dropout2(x)
        x = self.layer1(x)
        x = self.dropout3(x)
        x = self.layer2(x)
        x = self.dropout4(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout5(x)
        x = self.fc(x)

        return x

    def get_name(self):
        return type(self).__name__