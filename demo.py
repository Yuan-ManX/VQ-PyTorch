import torch

from vector_quantize import VectorQuantize


# 实例化一个矢量量化器（VectorQuantize），设置参数如下：
vq = VectorQuantize(
    dim = 256,               # 特征向量的维度为256
    codebook_size = 512,     # 码本大小为512，即有512个可学习的嵌入向量
    decay = 0.8,             # 指数移动平均的衰减率，值越低，字典更新速度越快
    commitment_weight = 1.   # 承诺损失的权重，用于平衡量化误差和码本更新
)


# 生成一个随机张量作为输入数据
# 形状为 (batch_size, sequence_length, feature_dim) = (1, 1024, 256)
x = torch.randn(1, 1024, 256)


# 对输入数据进行矢量量化
# 返回：
# - quantized: 量化后的嵌入向量，形状为 (1, 1024, 256)
# - indices: 量化时选择的码本索引，形状为 (1, 1024)
# - commit_loss: 承诺损失，用于训练时平衡量化误差和码本更新
quantized, indices, commit_loss = vq(x)  # 输出形状分别为 (1, 1024, 256), (1, 1024), (1)
