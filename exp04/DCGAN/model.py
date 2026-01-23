# model_gan.py
import torch
import torch.nn as nn

from exp04.DCGAN.config import config


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.z_dim = config.Z_DIM
        self.num_classes = config.NUM_CLASSES

        # 标签嵌入：将类别标签映射为嵌入向量
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)

        # 输入是 Z 和 Label 的拼接
        self.init_size = config.IMG_SIZE // 4  # 32 // 4 = 8
        self.l1 = nn.Sequential(
            nn.Linear(self.z_dim + self.num_classes, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),

            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出层
            nn.Conv2d(64, config.CHANNELS, 3, stride=1, padding=1),
            nn.Tanh()  # GAN 的输出通常使用 Tanh，范围 [-1, 1]
        )

    def forward(self, noise, labels):
        # 拼接噪声和标签
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)

        # 重塑为特征图 (Batch, 128, 8, 8)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 将标签嵌入并扩展为图像大小，作为额外的通道输入
        self.label_embedding = nn.Embedding(config.NUM_CLASSES, config.NUM_CLASSES)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        # self.model = nn.Sequential(
        #     # 输入通道 = 图片通道(3) + 标签通道(10，这里简化处理，通常做法有很多种)
        #     # 为了简单，我们在 forward 里把 label 变成全连接层输出再 reshape 拼上去
        #     # 或者更简单：直接在全连接层做条件判断，这里采用 PatchGAN 风格的 CNN
        #
        #     # 32x32 -> 16x16
        #     *discriminator_block(config.CHANNELS + config.NUM_CLASSES, 16, bn=False),
        #     # 16x16 -> 8x8
        #     *discriminator_block(16, 32),
        #     # 8x8 -> 4x4
        #     *discriminator_block(32, 64),
        #     # 4x4 -> 2x2
        #     *discriminator_block(64, 128),
        # )
        #
        # # 输出层：判断真假 (1维)
        # self.adv_layer = nn.Sequential(
        #     nn.Linear(128 * 2 * 2, 1),
        #     nn.Sigmoid()
        # )

        # 修改前：16 -> 32 -> 64 -> 128
        # 修改后：64 -> 128 -> 256 -> 512 (或者 32->64->128->256)

        # 示例：
        self.model = nn.Sequential(
            # 32x32 -> 16x16
            *discriminator_block(config.CHANNELS + config.NUM_CLASSES, 64, bn=False),
            # 16x16 -> 8x8
            *discriminator_block(64, 128),
            # 8x8 -> 4x4
            *discriminator_block(128, 256),
            # 4x4 -> 2x2
            *discriminator_block(256, 512),
        )
        # 记得修改最后的 Linear 层输入维度
        # 512 * 2 * 2
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 处理标签：将 Label 扩展成和 Image 一样大的 Feature Map
        # Label: (Batch) -> Embedding: (Batch, 10) -> Expand -> (Batch, 10, 32, 32)
        label_emb = self.label_embedding(labels)  # (N, 10)
        label_emb = label_emb.view(label_emb.size(0), config.NUM_CLASSES, 1, 1)
        label_img = label_emb.expand(label_emb.size(0), config.NUM_CLASSES, config.IMG_SIZE, config.IMG_SIZE)

        # 拼接图片和标签通道 (N, 13, 32, 32)
        d_in = torch.cat((img, label_img), 1)

        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# 权重初始化函数 (DCGAN 必须步骤)
def weights_init_normal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm2d' in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
