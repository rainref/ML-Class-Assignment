from model import resnet50
import os
import torch
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

# 模型网络结构可视化
net = resnet50()
print(net)

# 1. 使用torchsummary中的summary查看模型的输入输出形状、顺序结构，网络参数量，网络模型大小等信息
print('----torchsummary----')
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net.to(device)
summary(model, (3, 224, 224))    # 3是RGB通道数，即表示输入224 * 224的3通道的数据


# 2. 使用torchviz中的make_dot生成模型的网络结构，pdf图包括计算路径、网络各层的权重、偏移量
print('----torchviz----')
from torchviz import make_dot

X = torch.rand(size=(1, 3, 224, 224))    # 3是RGB通道数，即表示输入224 * 224的3通道的数据
Y = net(X)
vise = make_dot(Y, params=dict(net.named_parameters()))
vise.view()
#vise.render(filename='resnet50', format='pdf')  # 保存为pdf格式

# 3. 使用thop计算模型的FLOPs和参数量
print('----thop----')
from thop import profile
input = torch.randn(1, 3, 224, 224).to(device)
flops, params = profile(net, inputs=(input, ))
print("FLOPs: %.2f M" % (flops / 1e6))
print("Params: %.2f M" % (params / 1e6))

# 计算模型的参数量
total_params = sum(p.numel() for p in net.parameters())
print(f'Total parameters: {total_params / 1e6:.2f} M')
# 计算模型的可训练参数量
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Trainable parameters: {trainable_params / 1e6:.2f} M')
# 计算模型的不可训练参数量
non_trainable_params = total_params - trainable_params
print(f'Non-trainable parameters: {non_trainable_params / 1e6:.2f} M')
# 计算模型的层数
total_layers = sum(1 for p in net.parameters() if p.requires_grad and p.dim() > 1)
print(f'Total layers: {total_layers}')

# 4. 使用torchstat计算模型的统计信息
print('----torchstat----')
from torchstat import stat
stat(net, (3, 224, 224))    # 3是RGB通道数，即表示输入224 * 224的3通道的数据

# 5. 使用torchinfo计算模型的详细信息
print('----torchinfo----')
from torchinfo import summary
# 基本用法
# summary(net, input_size=(1, 3, 224, 224))

summary(
    net,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    verbose=1
)