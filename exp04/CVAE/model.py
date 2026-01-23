import torch
import torch.nn as nn


# ---  CVAE 模型定义 ---
class CVAE(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # --- Encoder (加深 + BatchNorm) ---
        self.encoder_conv = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),  # BN层
            nn.LeakyReLU(0.2),  # LeakyReLU 比 ReLU 更好

            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 8x8 -> 4x4
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.flatten_dim = 256 * 4 * 4
        self.label_embedding = nn.Embedding(self.num_classes, 128)

        self.fc_mu = nn.Linear(self.flatten_dim + 128, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim + 128, self.latent_dim)

        # --- Decoder ---
        self.decoder_input = nn.Linear(self.latent_dim + 128, self.flatten_dim)

        self.decoder_conv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()  # 输出层不需要 BN
        )

    def encode(self, x, labels):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten

        y = self.label_embedding(labels)

        combined_input = torch.cat([x, y], dim=1)
        mu = self.fc_mu(combined_input)
        logvar = self.fc_logvar(combined_input)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        y = self.label_embedding(labels)
        combined_input = torch.cat([z, y], dim=1)

        x = self.decoder_input(combined_input)
        x = x.view(x.size(0), 256, 4, 4)  # Unflatten
        x = self.decoder_conv(x)
        return x

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, labels)
        return reconstructed, mu, logvar
