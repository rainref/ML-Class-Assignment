# CVAE参数量
import torch
from torchinfo import summary

from exp04.CVAE.config import config
from exp04.CVAE.model import CVAE

cvae = CVAE(latent_dim=config.LATENT_DIM, num_classes=config.NUM_CLASSES).to(config.DEVICE)
# 基本用法
dummy_img = torch.randn(64, 3, 32, 32).to(config.DEVICE)
dummy_label = torch.randint(0, config.NUM_CLASSES, (64,)).to(config.DEVICE)

summary(
    cvae,
    input_data=[dummy_img, dummy_label],
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    verbose=1
)
