import torch
import torch.nn as nn
from einops import rearrange


# ==========================================
# 1. 核心组件：上采样与下采样
# ==========================================

class PatchMerging(nn.Module):
    """ Swin Transformer 的下采样模块 (Encoder用) """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ x: (B, H*W, C) """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # 拼图：将 2x2 的邻域拼成一个长向量
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    """ Swin-Unet 的上采样模块 (Decoder用) - 这是一个反向的 Patch Merging """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # 通过全连接层将通道数翻倍，以便 reshape 扩大分辨率
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """ x: (B, H*W, C) """
        H, W = self.input_resolution
        x = self.expand(x)  # B, H*W, 2C

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # Rearrange: 把通道维度拆开放到空间维度去 (PixelShuffle 的逻辑)
        # (B, H, W, 2C) -> (B, H, W, 2, C/2) -> (B, 2H, 2W, C/2)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


# ==========================================
# 2. 简化的 Swin Block (为了代码简洁，这里使用标准 Attention 示意)
# 注意：完整 Swin 需要 Shifted Window Attention，这里为了代码可运行性做了简化
# ==========================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinBlock(nn.Module):
    """ 一个基础的 Swin Transformer Block 单元 """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.norm1 = nn.LayerNorm(dim)

        # 这里为了演示简单，使用 PyTorch 自带的 MultiheadAttention
        # 在真实 Swin-Unet 中，这里应该是 WindowAttention + Shift logic
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        # Shortcut 1
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)  # Standard Attention for demo
        x = shortcut + x

        # Shortcut 2
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x


class BasicLayer(nn.Module):
    """ 一层由多个 Swin Block 组成的 Stage """

    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, window_size, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # 构建 Blocks
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
            for i in range(depth)
        ])

        # 下采样层 (Encoder 阶段使用)
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayerUp(nn.Module):
    """ Decoder 阶段的 Layer，包含 UpSampling """

    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, window_size, upsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
            for i in range(depth)
        ])

        # 上采样层 (Decoder 阶段使用)
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


# ==========================================
# 3. Swin-Unet 主模型
# ==========================================

class SwinUnet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=[2, 2, 2], num_heads=[3, 6, 12],
                 window_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]

        # 1. Patch Embedding (图片 -> Token)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            nn.LayerNorm(embed_dim)
        )  # Output: B, C, H*W -> Transpose later

        # ============ ENCODER (下采样) ============
        self.layers = nn.ModuleList()
        # Layer 0
        self.layers.append(BasicLayer(dim=embed_dim, output_dim=2 * embed_dim,
                                      input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                                      depth=depths[0], num_heads=num_heads[0], window_size=window_size,
                                      downsample=PatchMerging))
        # Layer 1
        self.layers.append(BasicLayer(dim=2 * embed_dim, output_dim=4 * embed_dim, input_resolution=(
            self.patches_resolution[0] // 2, self.patches_resolution[1] // 2),
                                      depth=depths[1], num_heads=num_heads[1], window_size=window_size,
                                      downsample=PatchMerging))
        # Layer 2 (Bottleneck)
        self.layers.append(BasicLayer(dim=4 * embed_dim, output_dim=8 * embed_dim, input_resolution=(
            self.patches_resolution[0] // 4, self.patches_resolution[1] // 4),
                                      depth=depths[2], num_heads=num_heads[2], window_size=window_size,
                                      downsample=None))

        # ============ DECODER (上采样) ============
        self.layers_up = nn.ModuleList()

        # Layer Up 0 (对应 Encoder Layer 1)
        self.layers_up.append(BasicLayerUp(dim=8 * embed_dim, output_dim=4 * embed_dim, input_resolution=(
            self.patches_resolution[0] // 4, self.patches_resolution[1] // 4),
                                           depth=depths[2], num_heads=num_heads[2], window_size=window_size,
                                           upsample=PatchExpand))

        # Layer Up 1 (对应 Encoder Layer 0)
        self.layers_up.append(BasicLayerUp(dim=4 * embed_dim, output_dim=2 * embed_dim, input_resolution=(
            self.patches_resolution[0] // 2, self.patches_resolution[1] // 2),
                                           depth=depths[1], num_heads=num_heads[1], window_size=window_size,
                                           upsample=PatchExpand))

        # Layer Up 2 (最后恢复)
        self.layers_up.append(BasicLayerUp(dim=2 * embed_dim, output_dim=embed_dim,
                                           input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                                           depth=depths[0], num_heads=num_heads[0], window_size=window_size,
                                           upsample=PatchExpand))

        self.norm = nn.LayerNorm(embed_dim)

        # Final Expand to original image size (4x upsample from patch size)
        self.final_up = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Input: (B, 3, 224, 224)
        x = self.patch_embed(x)  # (B, Embed_dim, H/4 * W/4)
        x = x.transpose(1, 2)  # (B, L, C)

        # Encoder 存储 Skip Connections
        skips = []

        # Encoder Flow
        # Block 0
        x = self.layers[0](x)  # Downsample occurred internally
        skips.append(x)  # Save for skip

        # Block 1
        x = self.layers[1](x)
        skips.append(x)

        # Block 2 (Bottleneck)
        x = self.layers[2](x)  # No downsample here

        # Decoder Flow (With Skip Connections)
        # Up Block 0
        x = self.layers_up[0](x)  # Upsample internally -> 4*Embed_dim

        # Skip Connection (Concatenation)
        # 注意：真实实现中 Swin-Unet 用的是 Linear Projection 来融合 Skip connection
        # 这里为了简化，我们假设维度匹配后直接加或者简单融合
        skip_x = skips.pop()  # Layer 1 output
        # 这里做一个简单的 Linear 融合演示 (Concat -> Linear)
        if x.shape == skip_x.shape:
            x = x + skip_x

            # Up Block 1
        x = self.layers_up[1](x)
        skip_x = skips.pop()  # Layer 0 output
        if x.shape == skip_x.shape:
            x = x + skip_x

        # Up Block 2
        x = self.layers_up[2](x)

        # Output
        x = self.norm(x)  # (B, L, C)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Final Upsampling to 224x224
        x = self.final_up(x)

        return torch.sigmoid(x)


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 模拟输入：Batch=1, Channel=3, Size=224x224
    dummy_input = torch.randn(1, 3, 224, 224)

    # 初始化模型
    model = SwinUnet(img_size=224, patch_size=4, in_chans=3, num_classes=1, embed_dim=96)

    # 前向传播
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # 应该输出 (1, 1, 224, 224)
