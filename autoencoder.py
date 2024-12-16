# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vector_quantize import VectorQuantize, Sequential


# 设置训练参数
lr = 3e-4 # 学习率为 3e-4
train_iter = 1000 # 训练迭代次数为 1000 次
num_codes = 256 # 码本的尺寸为 256
seed = 1234 # 随机种子为 1234
rotation_trick = True # 是否启用旋转技巧（数据增强的一种方式）
device = "cuda" if torch.cuda.is_available() else "cpu" # 如果有 GPU 可用，则使用 GPU；否则使用 CPU
 

# 定义一个简单的 VQ-AutoEncoder 模型
def SimpleVQAutoEncoder(**vq_kwargs):
    """
    定义一个简单的 VQ-AutoEncoder 模型。

    参数:
        **vq_kwargs: 向量量化层的其他关键字参数。

    返回:
        nn.Sequential: 包含编码器、向量量化层和解码器的顺序模型。
    """
    return Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # 编码器第一层：卷积层，输入通道数为 1，输出通道数为 16，卷积核大小为 3，步幅为 1，填充为 1
        nn.MaxPool2d(kernel_size=2, stride=2), # 最大池化层，池化核大小为 2，步幅为 2
        nn.GELU(), # GELU 激活函数
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 编码器第二层：卷积层，输入通道数为 16，输出通道数为 32，卷积核大小为 3，步幅为 1，填充为 1
        nn.MaxPool2d(kernel_size=2, stride=2), # 最大池化层，池化核大小为 2，步幅为 2
        VectorQuantize(dim=32, accept_image_fmap = True, **vq_kwargs), # 向量量化层，输入维度为 32，接受图像特征图
        nn.Upsample(scale_factor=2, mode="nearest"), # 上采样层，放大因子为 2，使用最近邻插值
        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), # 解码器第一层：卷积层，输入通道数为 32，输出通道数为 16，卷积核大小为 3，步幅为 1，填充为 1
        nn.GELU(), # GELU 激活函数
        nn.Upsample(scale_factor=2, mode="nearest"),  # 上采样层，放大因子为 2，使用最近邻插值
        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1), # 解码器第二层：卷积层，输入通道数为 16，输出通道数为 1，卷积核大小为 3，步幅为 1，填充为 1
    )


# 定义训练函数
def train(model, train_loader, train_iterations=1000, alpha=10):
    """
    训练模型。

    参数:
        model: 要训练的模型。
        train_loader: 训练数据加载器。
        train_iterations (int, 可选): 训练迭代次数。默认值为 1000。
        alpha (float, 可选): 损失函数中 cmt_loss 的权重。默认值为 10。
    """
    # 定义一个生成器，用于迭代数据集
    def iterate_dataset(data_loader):
        """
        生成器，用于无限迭代数据集。

        参数:
            data_loader: 数据加载器。

        Yields:
            Tuple[Tensor, Tensor]: 返回一个包含输入数据和标签的元组。
        """
        # 创建数据迭代器
        data_iter = iter(data_loader)
        while True:
            try:
                # 获取下一个批次的数据和标签
                x, y = next(data_iter)
            except StopIteration:
                # 如果迭代器耗尽，则重新创建迭代器
                data_iter = iter(data_loader)
                # 获取下一个批次的数据和标签
                x, y = next(data_iter)
            # 将数据和标签移动到设备，并返回
            yield x.to(device), y.to(device)
    
    # 使用 tqdm 显示进度条
    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        # 获取下一个批次的数据和标签
        x, _ = next(iterate_dataset(train_loader))

        # 前向传播，得到模型输出、量化索引和 cmt_loss
        out, indices, cmt_loss = model(x)
        # 将模型输出限制在 [-1, 1] 之间
        out = out.clamp(-1., 1.)

        # 计算重构损失，即输出与输入的绝对误差的平均值
        rec_loss = (out - x).abs().mean()
        # 计算总损失，包括重构损失和 cmt_loss
        (rec_loss + alpha * cmt_loss).backward()

        # 更新模型参数
        opt.step()
        # 更新进度条描述，显示当前的重构损失、cmt 损失和激活码本的百分比
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | " # 重构损失
            + f"cmt loss: {cmt_loss.item():.3f} | " # cmt 损失
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}" # 激活码本的百分比
        )


# 定义数据预处理转换
transform = transforms.Compose(
    [transforms.ToTensor(),  # 将图像转换为张量
     transforms.Normalize((0.5,), (0.5,))] # 对张量进行标准化，均值为 0.5，标准差为 0.5
) 


# 创建训练数据集的 DataLoader
train_dataset = DataLoader(
    datasets.FashionMNIST(
        root="~/data/fashion_mnist", # 数据集的根目录
        train=True,  # 加载训练集
        download=True,  # 如果数据集不存在，则自动下载
        transform=transform  # 应用数据预处理转换
    ),
    batch_size=256, # 每个批次的样本数量为 256
    shuffle=True, # 每个 epoch 开始时打乱数据顺序
)


# 设置随机种子，以确保结果可复现
torch.random.manual_seed(seed)


# 实例化 VQ-AutoEncoder 模型
model = SimpleVQAutoEncoder(
    codebook_size = num_codes, # 码本的尺寸，设置为 num_codes（256）
    rotation_trick = True, # 是否启用旋转技巧（数据增强的一种方式）
    straight_through = False # 是否启用直通梯度估计（Straight-Through Estimator）
).to(device) # 将模型移动到指定的设备（GPU 或 CPU）


# 定义优化器，使用 AdamW 优化算法
# 使用 AdamW 优化器，学习率为 lr（3e-4）
opt = torch.optim.AdamW(model.parameters(), lr=lr)


# 开始训练模型
train(model, train_dataset, train_iterations=train_iter)
