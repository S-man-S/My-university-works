from torch import nn, flatten


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 x 64 x 64 -> 64 x 16 x 16
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 64 x 16 x 16 -> 256 x 4 x 4
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 256 x 4 x 4 -> 1024 x 1 x 1
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linlayer = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.linlayer(x)
        return x


class LayerPart(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super().__init__()
        tmp_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, tmp_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(tmp_channels)
        self.conv2 = nn.Conv2d(tmp_channels, tmp_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(tmp_channels)
        self.conv3 = nn.Conv2d(tmp_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)

        return out


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(32, 128, 3, 1)
        self.layer2 = self.make_layer(128, 256, 4, 2)
        self.layer3 = self.make_layer(256, 512, 6, 2)
        self.layer4 = self.make_layer(512, 1024, 3, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = flatten(x, 1)
        x = self.fc(x)
        return x

    def make_layer(self, in_channels, out_channels, num_parts, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layer = [LayerPart(in_channels, out_channels, stride=stride, downsample=downsample)]
        for _ in range(1, num_parts):
            layer.append(LayerPart(out_channels, out_channels, stride=1, downsample=None))
        return nn.Sequential(*layer)


if __name__ == '__main__':
    model = CNN()
    print(model)
    exit(0)
