import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Saliency(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练ResNet18并拆分编码器
        resnet = models.resnet18(pretrained=pretrained)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64通道, 1/2
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64通道, 1/4
        self.encoder3 = resnet.layer2  # 128通道, 1/8
        self.encoder4 = resnet.layer3  # 256通道, 1/16
        self.encoder5 = resnet.layer4  # 512通道, 1/32

        # 解码器：上采样+特征融合
        self.decoder5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(64 + 64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.ConvTranspose2d(64 + 64, 1, kernel_size=2, stride=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器提取多尺度特征
        feat1 = self.encoder1(x)
        feat2 = self.encoder2(feat1)
        feat3 = self.encoder3(feat2)
        feat4 = self.encoder4(feat3)
        feat5 = self.encoder5(feat4)

        # 解码器融合与上采样
        dec5 = self.decoder5(feat5)
        fuse4 = torch.cat([dec5, feat4], dim=1)
        dec4 = self.decoder4(fuse4)

        fuse3 = torch.cat([dec4, feat3], dim=1)
        dec3 = self.decoder3(fuse3)

        fuse2 = torch.cat([dec3, feat2], dim=1)
        dec2 = self.decoder2(fuse2)

        fuse1 = torch.cat([dec2, feat1], dim=1)
        out = self.decoder1(fuse1)

        return self.sigmoid(out)
