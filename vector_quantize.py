from __future__ import annotations

from functools import partial, cache
from collections import namedtuple

import torch
from torch.nn import Module
from torch import nn, einsum, Tensor
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.amp import autocast

import einx
from einops import rearrange, repeat, reduce, pack, unpack

from typing import Callable


def exists(val):
    """
    检查一个值是否存在（不为 None）。

    参数:
        val: 需要检查的值。

    返回:
        bool: 如果 val 不为 None，则返回 True；否则返回 False。
    """
    return val is not None


def default(val, d):
    """
    返回可选值或默认值。

    参数:
        val: 需要检查的可选值。
        d: 默认值。

    返回:
        Any: 如果 val 存在，则返回 val；否则返回 d。
    """
    return val if exists(val) else d


def noop(*args, **kwargs):
    """
    空操作函数，不执行任何操作。

    参数:
        *args: 任意位置参数。
        **kwargs: 任意关键字参数。
    """
    pass


def identity(t):
    """
    恒等函数，返回输入值不变。

    参数:
        t: 输入值。

    返回:
        Any: 输入值 t。
    """
    return t


def l2norm(t, dim = -1,  eps = 1e-6):
    """
    对张量 t 进行 L2 归一化。

    参数:
        t (Tensor): 输入张量。
        dim (int, 可选): 需要归一化的维度。默认值为 -1。
        eps (float, 可选): 防止除以零的极小值。默认值为 1e-6。

    返回:
        Tensor: 归一化后的张量。
    """
    return F.normalize(t, p = 2, dim = dim, eps = eps)


def safe_div(num, den, eps = 1e-6):
    """
    安全除法函数，防止除以零。

    参数:
        num (Tensor): 分子张量。
        den (Tensor): 分母张量。
        eps (float, 可选): 防止除以零的极小值。默认值为 1e-6。

    返回:
        Tensor: 除法结果。
    """
    return num / den.clamp(min = eps)


def Sequential(*modules):
    """
    创建一个顺序模型，仅包含存在的模块。

    参数:
        *modules: 任意数量的 nn.Module 实例。

    返回:
        nn.Sequential: 包含所有存在模块的顺序模型。如果没有模块，则返回 None；如果只有一个模块，则返回该模块。
    """
    modules = [*filter(exists, modules)]
    if len(modules) == 0:
        return None
    elif len(modules) == 1:
        return modules[0]

    return nn.Sequential(*modules)


def cdist(x, y):
    """
    计算两个张量 x 和 y 之间的成对欧几里得距离。

    参数:
        x (Tensor): 第一个输入张量，形状为 (batch_size, n, d)。
        y (Tensor): 第二个输入张量，形状为 (batch_size, m, d)。

    返回:
        Tensor: 欧几里得距离矩阵，形状为 (batch_size, n, m)。
    """
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum') # 计算 x 的平方和，形状为 (batch_size, n)
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum') # 计算 y 的平方和，形状为 (batch_size, m)
    xy = einsum('b i d, b j d -> b i j', x, y) * -2 # 计算 x 和 y 的点积，形状为 (batch_size, n, m)
    # 计算欧几里得距离，并确保结果为非负数
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min = 0).sqrt()


def log(t, eps = 1e-20):
    """
    对张量 t 进行对数运算，并防止数值下溢。

    参数:
        t (Tensor): 输入张量。
        eps (float, 可选): 防止数值下溢的极小值。默认值为 1e-20。

    返回:
        Tensor: 对数运算后的张量。
    """
    return torch.log(t.clamp(min = eps))


def entropy(prob, eps = 1e-5):
    """
    计算概率分布的熵。

    参数:
        prob (Tensor): 概率分布张量，形状为 (batch_size, ..., n)。
        eps (float, 可选): 防止数值下溢的极小值。默认值为 1e-5。

    返回:
        Tensor: 熵，形状为 (batch_size, ...)。
    """
    return (-prob * log(prob, eps = eps)).sum(dim = -1)


def ema_inplace(old, new, decay):
    """
    对旧张量 old 进行指数移动平均（EMA）更新。

    参数:
        old (Tensor): 旧张量。
        new (Tensor): 新张量。
        decay (float): 衰减因子。
    """
    is_mps = str(old.device).startswith('mps:')

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


def pack_one(t, pattern):
    """
    将单个张量 t 按照指定的 pattern 打包，并返回一个解包函数。

    参数:
        t (Tensor): 需要打包的张量。
        pattern (str): 打包的模式。

    返回:
        Tuple[Tensor, callable]: 返回打包后的张量和解包函数。
    """
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        """
        解包函数，将打包后的张量 to_unpack 解包回原始形状。

        参数:
            to_unpack (Tensor): 需要解包的张量。
            unpack_pattern (Optional[str], 可选): 解包的 pattern。如果未提供，则使用原始的 pattern。

        返回:
            Tensor: 解包后的张量。
        """
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one


def lens_to_mask(lens, max_length):
    """
    根据序列长度 lens 生成掩码张量。

    参数:
        lens (Tensor): 序列长度张量，形状为 (batch_size,)。
        max_length (int): 最大序列长度。

    返回:
        Tensor: 掩码张量，形状为 (batch_size, max_length)。
    """
    # 生成序列索引张量
    seq = torch.arange(max_length, device = lens.device)
    # 生成掩码，标记有效位置
    return seq < lens[:, None]


def uniform_init(*shape):
    """
    使用 Kaiming 均匀初始化方法初始化一个张量。

    参数:
        *shape: 张量的形状参数。

    返回:
        Tensor: 初始化后的张量。
    """
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def gumbel_noise(t):
    """
    生成 Gumbel 噪声。

    参数:
        t (Tensor): 输入张量，用于确定噪声的形状。

    返回:
        Tensor: 与输入张量形状相同的 Gumbel 噪声。
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    dim = -1,
    training = True
):
    """
    使用 Gumbel-Softmax 对输入的对数概率进行采样。

    参数:
        logits (Tensor): 输入的对数概率张量。
        temperature (float, 可选): 温度参数。默认值为 1.0。
        stochastic (bool, 可选): 是否进行随机采样。默认值为 False。
        straight_through (bool, 可选): 是否使用直通梯度估计。默认值为 False。
        dim (int, 可选): 沿着哪个维度进行采样。默认值为 -1。
        training (bool, 可选): 是否在训练模式下进行采样。默认值为 True。

    返回:
        Tuple[Tensor, Tensor]: 返回采样索引和对应的 one-hot 编码。
    """
    # 获取数据类型和采样维度的大小
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        # 如果在训练模式下进行随机采样，并且温度大于 0，则应用 Gumbel-Softmax
        # 计算采样对数概率
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        # 否则，直接使用输入的对数概率
        sampling_logits = logits
    
    # 沿着指定维度进行 argmax 操作，得到采样索引
    ind = sampling_logits.argmax(dim = dim)
    # 将采样索引转换为 one-hot 编码
    one_hot = F.one_hot(ind, size).type(dtype)

    if not straight_through or temperature <= 0. or not training:
        # 如果不使用直通梯度估计，或者温度小于等于 0，或者不在训练模式下，则返回采样索引和 one-hot 编码
        return ind, one_hot
    
    # 计算 softmax 概率
    π1 = (logits / temperature).softmax(dim = dim)
    # 应用直通梯度估计，保留梯度信息
    one_hot = one_hot + π1 - π1.detach()

    # 返回采样索引和修正后的 one-hot 编码
    return ind, one_hot


def laplace_smoothing(x, n_categories, eps = 1e-5, dim = -1):
    """
    应用 Laplace 平滑到类别分布上。

    参数:
        x (Tensor): 输入的类别分布张量。
        n_categories (int): 类别数量。
        eps (float, 可选): 平滑因子。默认值为 1e-5。
        dim (int, 可选): 沿着哪个维度进行平滑。默认值为 -1。

    返回:
        Tensor: 平滑后的类别分布。
    """
    denom = x.sum(dim = dim, keepdim = True)
    return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num):
    """
    从样本中随机采样指定数量的向量。

    参数:
        samples (Tensor): 输入的样本张量。
        num (int): 需要采样的数量。

    返回:
        Tensor: 采样后的样本。
    """
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    """
    对批量样本进行批量采样。

    参数:
        samples (Tensor): 批量样本张量，形状为 (batch_size, ...)。
        num (int): 每个样本需要采样的数量。

    返回:
        Tensor: 批量采样后的样本，形状为 (batch_size, num, ...)。
    """
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)


def pad_shape(shape, size, dim = 0):
    """
    对形状列表进行填充。

    参数:
        shape (List[int]): 原始形状列表。
        size (int): 填充的大小。
        dim (int, 可选): 需要填充的维度。默认值为 0。

    返回:
        List[int]: 填充后的形状列表。
    """
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    """
    从多项式分布中采样。

    参数:
        total_count (int): 总计数。
        probs (Tensor): 多项式分布的概率张量。

    返回:
        Tensor: 采样结果，形状与 probs 相同。
    """
    device = probs.device
    probs = probs.cpu()

    # 创建一个新的张量，形状与 probs 相同，值为 total_count
    total_count = probs.new_full((), total_count)
    # 创建一个新的张量，形状与 probs 相同，值为 1
    remainder = probs.new_ones(())
    # 创建一个与 probs 形状相同的空张量，用于存储采样结果
    sample = torch.empty_like(probs, dtype = torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder) # 从二项分布中采样
        sample[i] = s # 存储采样结果
        total_count -= s # 更新总计数
        remainder -= p # 更新剩余概率

    # 确保总计数为 0
    assert total_count == 0, f'invalid total count {total_count}'
    # 返回采样结果，并移动回原始设备
    return sample.to(device)


def all_gather_sizes(x, dim):
    """
    收集所有进程在指定维度上的尺寸。

    参数:
        x (Tensor): 输入张量。
        dim (int): 需要收集尺寸的维度。

    返回:
        Tensor: 所有进程在该维度上的尺寸组成的张量。
    """
    # 获取指定维度的尺寸，并转换为长整型张量
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    # 创建一个列表，包含与 world_size 相同数量的空张量
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    # 在所有进程之间收集尺寸信息
    distributed.all_gather(all_sizes, size)
    # 将所有尺寸堆叠成一个张量并返回
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim = 0):
    """
    收集不同尺寸的张量。

    参数:
        x (Tensor): 输入张量。
        sizes (List[int]): 每个进程在指定维度上的尺寸列表。
        dim (int, 可选): 需要收集的维度。默认值为 0。

    返回:
        List[Tensor]: 收集到的所有张量列表。
    """
    # 获取当前进程的 rank
    rank = distributed.get_rank()
    # 初始化收集到的张量列表
    all_x = []

    for i, size in enumerate(sizes):
        # 如果当前进程的 rank 与索引 i 相同，则使用输入张量 x
        # 否则，创建一个与 x 形状相同但指定维度填充为 size 的空张量
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        # 从源进程 i 广播张量 t
        distributed.broadcast(t, src = i, async_op = True)
        # 将广播后的张量添加到列表中
        all_x.append(t)
    # 等待所有进程完成广播操作
    distributed.barrier()
    return all_x


def sample_vectors_distributed(local_samples, num):
    """
    在分布式环境中对本地样本进行采样，并收集所有样本。

    参数:
        local_samples (Tensor): 本地样本张量。
        num (int): 需要采样的总数量。

    返回:
        Tensor: 采样后的样本张量。
    """
    # 重塑本地样本张量的形状
    local_samples = rearrange(local_samples, '1 ... -> ...')

    # 获取当前进程的 rank
    rank = distributed.get_rank()
    # 收集所有进程在指定维度上的样本数量
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        # 如果当前进程是 rank 0，则根据所有进程的总样本数量进行多项式采样，确定每个进程需要采样的数量
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        # 否则，创建一个与 all_num_samples 形状相同的空张量
        samples_per_rank = torch.empty_like(all_num_samples)

    # 从 rank 0 广播每个进程需要采样的数量
    distributed.broadcast(samples_per_rank, src = 0)
    # 将采样数量转换为列表
    samples_per_rank = samples_per_rank.tolist()
    
    # 对本地样本进行采样
    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    # 收集所有进程采样后的样本
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    # 将所有样本连接成一个张量
    out = torch.cat(all_samples, dim = 0)
    # 重塑输出张量的形状并返回
    return rearrange(out, '... -> 1 ...')


def batched_bincount(x, *, minlength):
    """
    对批量数据进行单热编码（bincount 操作）。

    参数:
        x (Tensor): 输入张量，形状为 (batch_size, ...)。
        minlength (int): 输出张量的最小长度。

    返回:
        Tensor: 单热编码后的张量，形状为 (batch_size, minlength)。
    """
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    """
    K-Means 聚类算法。

    参数:
        samples (Tensor): 输入样本张量。
        num_clusters (int): 聚类数量。
        num_iters (int, 可选): 迭代次数。默认值为 10。
        use_cosine_sim (bool, 可选): 是否使用余弦相似度进行距离计算。默认值为 False。
        sample_fn (callable, 可选): 采样函数。默认使用 batched_sample_vectors。
        all_reduce_fn (callable, 可选): 全局归约函数。默认使用 noop。

    返回:
        Tuple[Tensor, Tensor]: 返回聚类中心张量和每个样本所属的聚类索引。
    """
    # 获取码本数量、维度、数据类型和设备
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    # 从样本中采样初始聚类中心
    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            # 如果使用余弦相似度，则计算样本与聚类中心的相似度
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            # 否则，计算欧几里得距离
            dists = -cdist(samples, means)

        # 根据距离分配每个样本到最近的聚类中心
        buckets = torch.argmax(dists, dim = -1)
        # 计算每个聚类的样本数量
        bins = batched_bincount(buckets, minlength = num_clusters)
        # 进行全局归约（如果需要）
        all_reduce_fn(bins)
        
        # 标记没有样本的聚类
        zero_mask = bins == 0
        # 将没有样本的聚类的数量设为 1，避免除以零
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        # 初始化新的聚类中心张量
        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        # 累加每个聚类中的样本
        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        # 计算新的聚类中心
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        # 进行全局归约（如果需要）
        all_reduce_fn(new_means)

        if use_cosine_sim:
            # 如果使用余弦相似度，则对新的聚类中心进行 L2 归一化
            new_means = l2norm(new_means)

        # 更新聚类中心，如果某个聚类没有样本，则保留原来的聚类中心
        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )
    
    # 返回最终的聚类中心和每个样本所属的聚类索引
    return means, bins


# rotation trick related

def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    """
    应用旋转技巧变换。

    参数:
        u (Tensor): 输入张量 u。
        q (Tensor): 输入张量 q。
        e (Tensor): 输入张量 e。

    返回:
        Tensor: 应用旋转技巧变换后的张量。
    """
    # 重塑 e 的形状为 (batch_size, 1, dim)
    e = rearrange(e, 'b d -> b 1 d')
    # 计算 u + q 的 L2 归一化，并分离计算图
    w = l2norm(u + q, dim = 1).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) + # 计算旋转项
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach()) # 计算缩放项
    )


def rotate_to(src, tgt):
    # rotation trick STE (https://arxiv.org/abs/2410.06424) to get gradients through VQ layer.
    """
    应用旋转技巧（Rotation Trick）以在 VQ 层中传递梯度。

    参数:
        src (Tensor): 源张量，形状为 (batch_size, ..., d)。
        tgt (Tensor): 目标张量，形状为 (batch_size, ..., d)。

    返回:
        Tensor: 应用旋转技巧后的张量。
    """
    # 对 src 和 tgt 进行打包，并保存解包函数
    src, inverse = pack_one(src, '* d') # src 的形状为 (batch_size, ..., d)
    tgt, _ = pack_one(tgt, '* d') # tgt 的形状为 (batch_size, ..., d)

    # 计算 src 和 tgt 的 L2 范数
    norm_src = src.norm(dim = -1, keepdim = True) # src 的 L2 范数，形状为 (batch_size, ..., 1)
    norm_tgt = tgt.norm(dim = -1, keepdim = True) # tgt 的 L2 范数，形状为 (batch_size, ..., 1)

    # 应用旋转技巧变换
    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src, norm_src), # src 的单位向量
        safe_div(tgt, norm_tgt), # tgt 的单位向量
        src # src 本身
    ).squeeze() # 旋转后的 tgt，形状为 (batch_size, ..., d)

    # 调整旋转后的 tgt 的范数，使其与 src 的范数成比例 
    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach() # 旋转后的张量

    # 返回解包后的旋转后的张量
    return inverse(rotated)


# distributed helpers

@cache # 缓存函数结果
def is_distributed():
    """
    判断当前环境是否为分布式环境。

    返回:
        bool: 如果是分布式环境且 world_size 大于 1，则返回 True；否则返回 False。
    """
    return distributed.is_initialized() and distributed.get_world_size() > 1


# regularization losses

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    """
    正交损失函数，用于正则化。

    参数:
        t (Tensor): 输入张量，形状为 (batch_size, ..., d)。

    返回:
        Tensor: 正交损失值。
    """
    # 获取张量的前两个维度的大小
    h, n = t.shape[:2]
    # 对输入张量进行 L2 归一化，形状为 (batch_size, ..., d)
    normed_codes = l2norm(t) 
    # 计算归一化后的张量的余弦相似度矩阵，形状为 (batch_size, ..., d, d)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    # 计算正交损失, 返回损失值
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)


# distance types

class EuclideanCodebook(Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True,
        manual_ema_update = False,
        affine_param = False,
        sync_affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9
    ):
        """
    欧几里得码本，用于量化输入向量。

    参数:
        dim (int): 特征维度。
        codebook_size (int): 码本中向量的数量。
        num_codebooks (int, 可选): 码本的数量。默认值为 1。
        kmeans_init (bool, 可选): 是否使用 K-Means 初始化码本。默认值为 False。
        kmeans_iters (int, 可选): K-Means 聚类的迭代次数。默认值为 10。
        sync_kmeans (bool, 可选): 是否在分布式环境中同步 K-Means。默认值为 True。
        decay (float, 可选): 指数移动平均（EMA）的衰减因子。默认值为 0.8。
        eps (float, 可选): 防止除以零的极小值。默认值为 1e-5。
        threshold_ema_dead_code (int, 可选): EMA 死码的阈值。默认值为 2。
        reset_cluster_size (int, 可选): 重置聚类大小的阈值。默认值为 threshold_ema_dead_code。
        use_ddp (bool, 可选): 是否使用分布式数据并行（Distributed Data Parallel）。默认值为 False。
        learnable_codebook (bool, 可选): 是否使码本可学习。默认值为 False。
        gumbel_sample (callable, 可选): Gumbel 采样函数。默认使用 gumbel_sample 函数。
        sample_codebook_temp (float, 可选): 采样码本的温度参数。默认值为 1.0。
        ema_update (bool, 可选): 是否使用 EMA 更新。默认值为 True。
        manual_ema_update (bool, 可选): 是否手动更新 EMA。默认值为 False。
        affine_param (bool, 可选): 是否使用仿射参数。默认值为 False。
        sync_affine_param (bool, 可选): 是否在分布式环境中同步仿射参数。默认值为 False。
        affine_param_batch_decay (float, 可选): 仿射参数的批次衰减因子。默认值为 0.99。
        affine_param_codebook_decay (float, 可选): 仿射参数的码本衰减因子。默认值为 0.9。
    """
        super().__init__()
        # 设置输入转换函数为恒等函数
        self.transform_input = identity

        # 指数移动平均（EMA）的衰减因子
        self.decay = decay
        # 是否使用 EMA 更新
        self.ema_update = ema_update
        # 是否手动更新 EMA
        self.manual_ema_update = manual_ema_update

        # 初始化码本嵌入
        # 选择初始化函数
        init_fn = uniform_init if not kmeans_init else torch.zeros
        # 初始化码本嵌入
        embed = init_fn(num_codebooks, codebook_size, dim)

        # 码本中向量的数量
        self.codebook_size = codebook_size
        # 码本的数量
        self.num_codebooks = num_codebooks

        # K-Means 聚类的迭代次数
        self.kmeans_iters = kmeans_iters
        # 防止除以零的极小值
        self.eps = eps
        # EMA 死码的阈值
        self.threshold_ema_dead_code = threshold_ema_dead_code
        # 重置聚类大小的阈值
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        # 确保 gumbel_sample 是一个可调用的函数
        assert callable(gumbel_sample)
        # Gumbel 采样函数
        self.gumbel_sample = gumbel_sample
        # 采样码本的温度参数
        self.sample_codebook_temp = sample_codebook_temp

        # 检查是否在分布式环境中使用 K-Means 初始化多个码本
        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'

        # 选择采样函数
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors

        self.replace_sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors

        # 选择全局归约函数
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        # 注册缓冲区
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        # 聚类大小
        self.register_buffer('cluster_size', torch.ones(num_codebooks, codebook_size))
        # 码本嵌入的平均值
        self.register_buffer('embed_avg', embed.clone())

        # 是否使码本可学习
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            # 使码本嵌入可学习
            self.embed = nn.Parameter(embed)
        else:
            # 注册码本嵌入
            self.register_buffer('embed', embed)

        # affine related params（仿射参数相关）

        # 是否使用仿射参数
        self.affine_param = affine_param
        # 是否在分布式环境中同步仿射参数
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            # 如果不使用仿射参数，则返回
            return

        # 仿射参数的批次衰减因子
        self.affine_param_batch_decay = affine_param_batch_decay
        # 仿射参数的码本衰减因子
        self.affine_param_codebook_decay = affine_param_codebook_decay

        # 批次均值
        self.register_buffer('batch_mean', None)
        # 批次方差
        self.register_buffer('batch_variance', None)

        # 码本均值是否需要初始化
        self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
        # 码本均值
        self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
        # 码本方差是否需要初始化
        self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
        # 码本方差
        self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        """
        初始化码本嵌入。

        参数:
            data (Tensor): 输入数据。
            mask (Optional[Tensor], 可选): 数据掩码。默认值为 None。
        """
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            # 根据掩码重塑数据
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        # 使用 K-Means 进行聚类
        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        # 计算嵌入的总和
        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        # 复制嵌入到嵌入缓冲区
        self.embed.data.copy_(embed)
        # 复制嵌入的总和到嵌入平均值缓冲区
        self.embed_avg.data.copy_(embed_sum)
        # 复制聚类大小到聚类大小缓冲区
        self.cluster_size.data.copy_(cluster_size)
        # 标记为已初始化
        self.initted.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        """
        使用衰减因子更新缓冲区中的值。

        参数:
            buffer_name (str): 缓冲区的名称。
            new_value (Tensor): 新的值。
            decay (float): 衰减因子。
        """
        old_value = getattr(self, buffer_name)

        # 获取是否需要初始化
        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            # 如果需要初始化，则标记为已初始化
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not exists(old_value) or needs_init:
            # 如果旧值不存在或需要初始化，则注册新的值
            self.register_buffer(buffer_name, new_value.detach())

            return
        # 计算新的值，使用衰减因子
        value = old_value * decay + new_value.detach() * (1 - decay)
        # 更新缓冲区中的值
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embed, mask = None):
        """
        更新仿射参数，包括批次均值和方差以及码本均值和方差。

        参数:
            data (Tensor): 输入数据。
            embed (Tensor): 嵌入数据。
            mask (Optional[Tensor], 可选): 数据掩码。默认值为 None。
        """
        # 确保仿射参数已启用
        assert self.affine_param
        # 定义方差函数
        var_fn = partial(torch.var, unbiased = False)

        # calculate codebook mean and variance
        # 计算码本均值和方差
        # 重塑嵌入数据的形状
        embed = rearrange(embed, 'h ... d -> h (...) d')

        if self.training:
            # 在训练模式下，使用 EMA 更新码本均值和方差
            self.update_with_decay('codebook_mean', reduce(embed, 'h n d -> h 1 d', 'mean'), self.affine_param_codebook_decay)
            self.update_with_decay('codebook_variance', reduce(embed, 'h n d -> h 1 d', var_fn), self.affine_param_codebook_decay)

        # prepare batch data, which depends on whether it has masking
        # 准备批次数据，根据是否使用掩码
        # 重塑输入数据的形状
        data = rearrange(data, 'h ... d -> h (...) d')

        if exists(mask):
            c = data.shape[0]
            # 如果存在掩码，则应用掩码
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        # calculate batch mean and variance
        # 计算批次均值和方差

        if not self.sync_affine_param:
            # 如果不同步仿射参数，则直接使用 EMA 更新批次均值和方差
            self.update_with_decay('batch_mean', reduce(data, 'h n d -> h 1 d', 'mean'), self.affine_param_batch_decay)
            self.update_with_decay('batch_variance', reduce(data, 'h n d -> h 1 d', var_fn), self.affine_param_batch_decay)
            return

        # 获取向量数量、设备和数据类型
        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

        # number of vectors, for denominator
        # 计算分布式均值
        # 创建向量数量的张量
        num_vectors = torch.tensor([num_vectors], device = device, dtype = dtype)
        # 在分布式环境中进行归约
        distributed.all_reduce(num_vectors)

        # calculate distributed mean
        # 计算批次数据的总和
        batch_sum = reduce(data, 'h n d -> h 1 d', 'sum')
        # 在分布式环境中进行归约
        distributed.all_reduce(batch_sum)
        # 计算批次均值
        batch_mean = batch_sum / num_vectors

        # 更新批次均值
        self.update_with_decay('batch_mean', batch_mean, self.affine_param_batch_decay)

        # calculate distributed variance
        # 计算分布式方差
        # 计算方差的分子部分
        variance_numer = reduce((data - batch_mean) ** 2, 'h n d -> h 1 d', 'sum')
        # 在分布式环境中进行归约
        distributed.all_reduce(variance_numer)
        # 计算批次方差
        batch_variance = variance_numer / num_vectors
        # 更新批次方差
        self.update_with_decay('batch_variance', batch_variance, self.affine_param_batch_decay)

    def replace(self, batch_samples, batch_mask):
        """
        替换码本中的样本。

        参数:
            batch_samples (Tensor): 批次样本。
            batch_mask (Tensor): 批次掩码。
        """
        for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
            # 采样替换样本
            sampled = self.replace_sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            # 重塑采样后的样本
            sampled = rearrange(sampled, '1 ... -> ...')

            # 替换嵌入数据
            self.embed.data[ind][mask] = sampled
            # 重置聚类大小
            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            # 更新嵌入平均值
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        """
        过期码本中的死码。

        参数:
            batch_samples (Tensor): 批次样本。
        """
        if self.threshold_ema_dead_code == 0:
            # 如果死码阈值为 0，则不进行过期操作
            return

        # 判断哪些码是死码
        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            # 如果没有死码，则返回
            return

        # 重塑批次样本的形状
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        # 替换死码
        self.replace(batch_samples, batch_mask = expired_codes)

    def update_ema(self):
        """
        更新 EMA（指数移动平均）嵌入。
        """
        # 应用 Laplace 平滑
        cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

        # 归一化嵌入平均值
        embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
        # 更新嵌入数据
        self.embed.data.copy_(embed_normalized)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False,
        codebook_transform_fn: Callable | None = None
    ):
        """
        前向传播函数，用于量化输入张量。

        参数:
            x (Tensor): 输入张量。
            sample_codebook_temp (Optional[float], 可选): 采样码本的温度参数。默认使用初始化时的值。
            mask (Optional[Tensor], 可选): 数据掩码。默认值为 None。
            freeze_codebook (bool, 可选): 是否冻结码本。默认值为 False。
            codebook_transform_fn (Optional[Callable], 可选): 码本转换函数。默认值为 None。

        返回:
            Tuple[Tensor, Tensor, Tensor]: 返回量化后的张量、嵌入索引和张量的距离。
        """
        # 判断是否需要调整码本的维度
        needs_codebook_dim = x.ndim < 4
        # 设置采样码本的温度参数
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        # 将输入张量转换为浮点型
        x = x.float()

        if needs_codebook_dim:
            # 如果需要调整维度，则重塑张量形状
            x = rearrange(x, '... -> 1 ...')

        # 获取数据类型
        dtype = x.dtype
        # 展平输入张量，并返回解包函数
        flatten, unpack_one = pack_one(x, 'h * d')

        if exists(mask):
            # 如果存在掩码，则重复掩码以匹配展平后的张量形状
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        # 初始化码本嵌入
        self.init_embed_(flatten, mask = mask)

        if self.affine_param:
            # 如果使用仿射参数，则更新仿射参数
            self.update_affine(flatten, self.embed, mask = mask)

        # affine params（仿射参数）

        if self.affine_param:
            # 计算码本的标准差和批次的标准差
            codebook_std = self.codebook_variance.clamp(min = 1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min = 1e-5).sqrt()
            # 应用仿射变换
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

        # get maybe learnable codes
        # 如果码本可学习，则使用可学习的码本；否则，使用不可学习的码本
        embed = self.embed if self.learnable_codebook else self.embed.detach()

        # handle maybe implicit neural codebook
        # and calculate distance
        # 处理可能隐式的神经码本
        # 并计算距离

        if exists(codebook_transform_fn):
            # 如果存在码本转换函数，则应用转换
            transformed_embed = codebook_transform_fn(embed)
            transformed_embed = rearrange(transformed_embed, 'h b n c d -> h (b n) c d')
            broadcastable_input = rearrange(flatten, '... d -> ... 1 d')

            # 计算距离
            dist = -F.pairwise_distance(broadcastable_input, transformed_embed)
        else:
            # 如果不存在码本转换函数，则直接计算距离
            dist = -cdist(flatten, embed)

        # sample or argmax depending on temperature
        # 根据温度参数进行采样或 argmax

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)
        # 解包嵌入索引
        embed_ind = unpack_one(embed_ind, 'h *')

        if exists(codebook_transform_fn):
            # 解包转换后的嵌入
            transformed_embed = unpack_one(transformed_embed, 'h * c d')

        if self.training:
            # 解包 one-hot 编码
            unpacked_onehot = unpack_one(embed_onehot, 'h * c')

            if exists(codebook_transform_fn):
                # 如果存在码本转换函数，则计算量化后的张量
                quantize = einsum('h b n c, h b n c d -> h b n d', unpacked_onehot, transformed_embed)
            else:
                # 否则，使用嵌入进行量化
                quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)

        else:
            if exists(codebook_transform_fn):
                # 如果存在码本转换函数，则使用嵌入索引进行索引
                quantize = einx.get_at('h b n [c] d, h b n -> h b n d', transformed_embed, embed_ind)
            else:
                # 否则，使用嵌入进行索引
                quantize = einx.get_at('h [c] d, h b n -> h b n d', embed, embed_ind)

        if self.training and self.ema_update and not freeze_codebook:
            # 如果在训练模式下，使用 EMA 更新，并且不冻结码本

            if self.affine_param:
                # 如果使用仿射参数，则反转仿射变换
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

            if exists(mask):
                # 应用掩码
                embed_onehot[~mask] = 0.

            # 计算聚类大小
            cluster_size = embed_onehot.sum(dim = 1)

            # 全局归约聚类大小
            self.all_reduce_fn(cluster_size)
            # 更新 EMA 聚类大小
            ema_inplace(self.cluster_size.data, cluster_size, self.decay)

            # 计算嵌入的总和
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            # 确保嵌入总和是连续的
            embed_sum = embed_sum.contiguous()
            # 全局归约嵌入总和
            self.all_reduce_fn(embed_sum)

            # 更新 EMA 嵌入平均值
            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            if not self.manual_ema_update:
                # 更新 EMA 嵌入
                self.update_ema()
                # 过期死码
                self.expire_codes_(x)

        if needs_codebook_dim:
            # 如果需要调整码本维度，则重塑量化后的张量和嵌入索引
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        # 解包距离
        dist = unpack_one(dist, 'h * d')

        # 返回量化后的张量、嵌入索引和张量的距离
        return quantize, embed_ind, dist


class CosineSimCodebook(Module):
    """
    基于余弦相似度的码本，用于量化输入向量。

    参数:
        dim (int): 特征维度。
        codebook_size (int): 码本中向量的数量。
        num_codebooks (int, 可选): 码本的数量。默认值为 1。
        kmeans_init (bool, 可选): 是否使用 K-Means 初始化码本。默认值为 False。
        kmeans_iters (int, 可选): K-Means 聚类的迭代次数。默认值为 10。
        sync_kmeans (bool, 可选): 是否在分布式环境中同步 K-Means。默认值为 True。
        decay (float, 可选): 指数移动平均（EMA）的衰减因子。默认值为 0.8。
        eps (float, 可选): 防止除以零的极小值。默认值为 1e-5。
        threshold_ema_dead_code (int, 可选): EMA 死码的阈值。默认值为 2。
        reset_cluster_size (int, 可选): 重置聚类大小的阈值。默认值为 threshold_ema_dead_code。
        use_ddp (bool, 可选): 是否使用分布式数据并行（Distributed Data Parallel）。默认值为 False。
        learnable_codebook (bool, 可选): 是否使码本可学习。默认值为 False。
        gumbel_sample (callable, 可选): Gumbel 采样函数。默认使用 gumbel_sample 函数。
        sample_codebook_temp (float, 可选): 采样码本的温度参数。默认值为 1.0。
        ema_update (bool, 可选): 是否使用 EMA 更新。默认值为 True。
        manual_ema_update (bool, 可选): 是否手动更新 EMA。默认值为 False。
    """
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True,
        manual_ema_update = False
    ):
        super().__init__()
        # 设置输入转换函数为 L2 归一化
        self.transform_input = l2norm

        # 是否使用 EMA 更新
        self.ema_update = ema_update
        # 是否手动更新 EMA
        self.manual_ema_update = manual_ema_update

        # 指数移动平均（EMA）的衰减因子
        self.decay = decay

        if not kmeans_init:
            # 如果不使用 K-Means 初始化，则使用均匀初始化并归一化
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            # 否则，使用全零初始化
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        # 码本中向量的数量
        self.codebook_size = codebook_size
        # 码本的数量
        self.num_codebooks = num_codebooks

        # K-Means 聚类的迭代次数
        self.kmeans_iters = kmeans_iters
        # 防止除以零的极小值
        self.eps = eps
        # EMA 死码的阈值
        self.threshold_ema_dead_code = threshold_ema_dead_code
        # 重置聚类大小的阈值
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        # 确保 gumbel_sample 是一个可调用的函数
        assert callable(gumbel_sample)
        # Gumbel 采样函数
        self.gumbel_sample = gumbel_sample
        # 采样码本的温度参数
        self.sample_codebook_temp = sample_codebook_temp

        # 选择采样函数
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors

        self.replace_sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors

        # 选择全局归约函数
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        # 注册缓冲区
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))  # 是否初始化
        self.register_buffer('cluster_size', torch.ones(num_codebooks, codebook_size)) # 聚类大小
        self.register_buffer('embed_avg', embed.clone()) # 码本嵌入的平均值

        # 是否使码本可学习
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            # 使码本嵌入可学习
            self.embed = nn.Parameter(embed)
        else:
            # 注册码本嵌入
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        """
        初始化码本嵌入。

        参数:
            data (Tensor): 输入数据。
            mask (Optional[Tensor], 可选): 数据掩码。默认值为 None。
        """
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c) # 根据掩码重塑数据

        # 使用 K-Means 进行聚类
        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True, # 使用余弦相似度
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        # 计算嵌入的总和
        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        self.embed.data.copy_(embed) # 复制嵌入到嵌入缓冲区 
        self.embed_avg.data.copy_(embed_sum) # 复制嵌入的总和到嵌入平均值缓冲区
        self.cluster_size.data.copy_(cluster_size) # 复制聚类大小到聚类大小缓冲区
        self.initted.data.copy_(torch.Tensor([True])) # 标记为已初始化

    def replace(self, batch_samples, batch_mask):
        """
        替换码本中的样本。

        参数:
            batch_samples (Tensor): 批次样本。
            batch_mask (Tensor): 批次掩码。
        """
        # 对批次样本进行 L2 归一化
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
            # 使用替换样本函数进行采样，mask.sum().item() 表示需要采样的数量
            sampled = self.replace_sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            # 重塑采样后的样本形状
            sampled = rearrange(sampled, '1 ... -> ...')

            # 替换嵌入数据
            self.embed.data[ind][mask] = sampled
            # 更新嵌入平均值
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size
            # 重置聚类大小
            self.cluster_size.data[ind][mask] = self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        """
        过期码本中的死码。

        参数:
            batch_samples (Tensor): 批次样本。
        """
        if self.threshold_ema_dead_code == 0:
            return # 如果死码阈值为 0，则不进行过期操作

        # 判断哪些码是死码
        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return # 如果没有死码，则返回

        # 重塑批次样本的形状
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        # 替换死码
        self.replace(batch_samples, batch_mask = expired_codes)

    def update_ema(self):
        """
        更新 EMA（指数移动平均）嵌入。
        """
        # 应用 Laplace 平滑
        cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

        # 计算归一化的嵌入平均值
        embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
        # 对嵌入平均值进行 L2 归一化
        embed_normalized = l2norm(embed_normalized)

        # 更新嵌入数据
        self.embed.data.copy_(embed_normalized)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False,
        codebook_transform_fn: Callable | None = None
    ):
        """
        前向传播函数，用于量化输入张量。

        参数:
            x (Tensor): 输入张量。
            sample_codebook_temp (Optional[float], 可选): 采样码本的温度参数。默认使用初始化时的值。
            mask (Optional[Tensor], 可选): 数据掩码。默认值为 None。
            freeze_codebook (bool, 可选): 是否冻结码本。默认值为 False。
            codebook_transform_fn (Optional[Callable], 可选): 码本转换函数。默认值为 None。

        返回:
            Tuple[Tensor, Tensor, Tensor]: 返回量化后的张量、嵌入索引和张量的距离。
        """
        # 判断是否需要调整码本的维度
        needs_codebook_dim = x.ndim < 4
        # 设置采样码本的温度参数
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        # 将输入张量转换为浮点型
        x = x.float()

        if needs_codebook_dim:
            # 如果需要调整维度，则重塑张量形状
            x = rearrange(x, '... -> 1 ...')

        # 获取数据类型
        dtype = x.dtype

        # 展平输入张量，并返回解包函数
        flatten, unpack_one = pack_one(x, 'h * d')

        if exists(mask):
            # 如果存在掩码，则重复掩码以匹配展平后的张量形状
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        # 初始化码本嵌入
        self.init_embed_(flatten, mask = mask)

        # 获取码本嵌入，如果码本可学习，则使用可学习的码本；否则，使用不可学习的码本
        embed = self.embed if self.learnable_codebook else self.embed.detach()

        # handle maybe implicit neural codebook
        # and compute cosine sim distance
        # 处理可能隐式的神经码本
        # 并计算余弦相似度距离

        if exists(codebook_transform_fn):
            # 如果存在码本转换函数，则应用转换并归一化
            transformed_embed = codebook_transform_fn(embed)
            transformed_embed = rearrange(transformed_embed, 'h b n c d -> h (b n) c d')
            transformed_embed = l2norm(transformed_embed)

            # 计算余弦相似度距离
            dist = einsum('h n d, h n c d -> h n c', flatten, transformed_embed)
        else:
            # 如果不存在码本转换函数，则直接计算余弦相似度距离
            dist = einsum('h n d, h c d -> h n c', flatten, embed)

        # 根据温度参数进行采样或 argmax
        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)
        # 解包嵌入索引
        embed_ind = unpack_one(embed_ind, 'h *')

        if exists(codebook_transform_fn):
            # 解包转换后的嵌入
            transformed_embed = unpack_one(transformed_embed, 'h * c d')

        if self.training:
            # 解包 one-hot 编码
            unpacked_onehot = unpack_one(embed_onehot, 'h * c')

            if exists(codebook_transform_fn):
                # 如果存在码本转换函数，则计算量化后的张量
                quantize = einsum('h b n c, h b n c d -> h b n d', unpacked_onehot, transformed_embed)
            else:
                # 否则，使用嵌入进行量化
                quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)

        else:
            if exists(codebook_transform_fn):
                # 如果存在码本转换函数，则使用嵌入索引进行索引
                quantize = einx.get_at('h b n [c] d, h b n -> h b n d', transformed_embed, embed_ind)
            else:
                # 否则，使用嵌入进行索引
                quantize = einx.get_at('h [c] d, h b n -> h b n d', embed, embed_ind)

        if self.training and self.ema_update and not freeze_codebook:
            # 如果在训练模式下，使用 EMA 更新，并且不冻结码本
            if exists(mask):
                # 应用掩码
                embed_onehot[~mask] = 0.

            # 计算聚类大小
            bins = embed_onehot.sum(dim = 1)
            # 全局归约聚类大小
            self.all_reduce_fn(bins)
            
            # 更新 EMA 聚类大小
            ema_inplace(self.cluster_size.data, bins, self.decay)

            # 计算嵌入的总和
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            # 确保嵌入总和是连续的
            embed_sum = embed_sum.contiguous()
            # 全局归约嵌入总和
            self.all_reduce_fn(embed_sum)

            # 更新 EMA 嵌入平均值
            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            if not self.manual_ema_update:
                self.update_ema() # 更新 EMA 嵌入
                self.expire_codes_(x) 

        if needs_codebook_dim:
            # 如果需要调整码本维度，则重塑量化后的张量和嵌入索引
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))
        
        # 解包距离
        dist = unpack_one(dist, 'h * d')
        # 返回量化后的张量、嵌入索引和张量的距离
        return quantize, embed_ind, dist


# main class

# 定义损失分解的命名元组，用于存储不同类型的损失
LossBreakdown = namedtuple('LossBreakdown', [
    'commitment', # 承诺损失：用于确保编码器输出与码本向量之间的匹配
    'codebook_diversity', # 码本多样性损失：鼓励码本中不同向量之间的多样性
    'orthogonal_reg', # 正交正则化损失：用于正则化码本向量，使其彼此正交
    'inplace_optimize', # 就地优化损失：用于在原地优化过程中计算的损失
])


class VectorQuantize(Module):
    """
    向量量化模块，用于将输入向量量化到离散的码本空间中。

    参数:
        dim (int): 输入数据的维度。
        codebook_size (int): 码本中向量的数量。
        codebook_dim (int, 可选): 码本向量的维度。如果未提供，则默认为 dim。
        heads (int, 可选): 注意力头的数量。默认值为 1。
        separate_codebook_per_head (bool, 可选): 是否为每个注意力头使用独立的码本。默认值为 False。
        decay (float, 可选): 指数移动平均（EMA）的衰减因子。默认值为 0.8。
        eps (float, 可选): 防止除以零的极小值。默认值为 1e-5。
        freeze_codebook (bool, 可选): 是否冻结码本。默认值为 False。
        kmeans_init (bool, 可选): 是否使用 K-Means 初始化码本。默认值为 False。
        kmeans_iters (int, 可选): K-Means 聚类的迭代次数。默认值为 10。
        sync_kmeans (bool, 可选): 是否在分布式环境中同步 K-Means。默认值为 True。
        use_cosine_sim (bool, 可选): 是否使用余弦相似度进行距离计算。默认值为 False。
        layernorm_after_project_in (bool, 可选): 是否在投影后应用层归一化。默认值为 False。
        threshold_ema_dead_code (int, 可选): EMA 死码的阈值。默认值为 0。
        channel_last (bool, 可选): 是否将通道维度放在最后。默认值为 True。
        accept_image_fmap (bool, 可选): 是否接受图像特征图。默认值为 False。
        commitment_weight (float, 可选): 承诺损失的权重。默认值为 1.0。
        commitment_use_cross_entropy_loss (bool, 可选): 是否使用交叉熵损失作为承诺损失。默认值为 False。
        orthogonal_reg_weight (float, 可选): 正交正则化损失的权重。默认值为 0.0。
        orthogonal_reg_active_codes_only (bool, 可选): 是否仅对活跃码本应用正交正则化。默认值为 False。
        orthogonal_reg_max_codes (int, 可选): 正交正则化损失中最大码本数量。默认值为 None。
        codebook_diversity_loss_weight (float, 可选): 码本多样性损失的权重。默认值为 0.0。
        codebook_diversity_temperature (float, 可选): 码本多样性损失的温度参数。默认值为 100.0。
        stochastic_sample_codes (bool, 可选): 是否进行随机采样。默认值为 False。
        sample_codebook_temp (float, 可选): 采样码本的温度参数。默认值为 1.0。
        straight_through (bool, 可选): 是否使用直通梯度估计。默认值为 False。
        rotation_trick (bool, 可选): 是否使用旋转技巧传递梯度。默认值为 True。
        sync_codebook (bool, 可选): 是否同步码本。默认值为 None。
        sync_affine_param (bool, 可选): 是否同步仿射参数。默认值为 False。
        ema_update (bool, 可选): 是否使用 EMA 更新。默认值为 True。
        manual_ema_update (bool, 可选): 是否手动更新 EMA。默认值为 False。
        learnable_codebook (bool, 可选): 是否使码本可学习。默认值为 False。
        in_place_codebook_optimizer (Optimizer, 可选): 用于更新可学习码本的优化器。默认值为 None。
        manual_in_place_optimizer_update (bool, 可选): 是否手动更新就地优化器。默认值为 False。
        affine_param (bool, 可选): 是否使用仿射参数。默认值为 False。
        affine_param_batch_decay (float, 可选): 仿射参数的批次衰减因子。默认值为 0.99。
        affine_param_codebook_decay (float, 可选): 仿射参数的码本衰减因子。默认值为 0.9。
        sync_update_v (float, 可选): 控制同步更新规则中乐观与悲观更新的参数。默认值为 0.0。
        return_zeros_for_masked_padding (bool, 可选): 是否对掩码填充返回零。默认值为 True。
    """
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim = None,
        heads = 1,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        freeze_codebook = False,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        use_cosine_sim = False,
        layernorm_after_project_in = False, # proposed by @SaltyChtao here https://github.com/lucidrains/vector-quantize-pytorch/issues/26#issuecomment-1324711561
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        commitment_use_cross_entropy_loss = False,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        codebook_diversity_loss_weight = 0.,
        codebook_diversity_temperature = 100.,
        stochastic_sample_codes = False,
        sample_codebook_temp = 1.,
        straight_through = False,
        rotation_trick = True,  # Propagate grads through VQ layer w/ rotation trick: https://arxiv.org/abs/2410.06424 by @cfifty
        sync_codebook = None,
        sync_affine_param = False,
        ema_update = True,
        manual_ema_update = False,
        learnable_codebook = False,
        in_place_codebook_optimizer: Callable[..., Optimizer] = None, # Optimizer used to update the codebook embedding if using learnable_codebook
        manual_in_place_optimizer_update = False,
        affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9,
        sync_update_v = 0., # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
        return_zeros_for_masked_padding = True
    ):
        super().__init__()
        # 存储输入数据的维度
        self.dim = dim
        # 存储注意力头的数量
        self.heads = heads
        # 存储是否为每个注意力头使用独立的码本
        self.separate_codebook_per_head = separate_codebook_per_head

        # 如果未提供 codebook_dim，则默认为 dim
        codebook_dim = default(codebook_dim, dim)
        # 计算码本输入维度
        codebook_input_dim = codebook_dim * heads

        # 判断是否需要进行投影
        requires_projection = codebook_input_dim != dim

        # 定义投影层
        self.project_in = Sequential(
            nn.Linear(dim, codebook_input_dim),
            nn.LayerNorm(codebook_input_dim) if layernorm_after_project_in else None
        ) if requires_projection else nn.Identity()

        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        # 存储是否需要进行投影
        self.has_projections = requires_projection

        # 存储防止除以零的极小值
        self.eps = eps

        # 判断是否具有承诺损失
        self.has_commitment_loss = commitment_weight > 0.
        # 存储承诺损失的权重
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss # whether to use cross entropy loss to codebook as commitment loss

        self.learnable_codebook = learnable_codebook

        # 判断是否具有码本正交损失
        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0.
        # 存储是否具有码本正交损失
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        # 存储正交正则化损失的权重
        self.orthogonal_reg_weight = orthogonal_reg_weight
        # 是否仅对活跃码本应用正交正则化
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        # 正交正则化损失中最大码本数量
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        # 判断是否具有码本多样性损失
        has_codebook_diversity_loss = codebook_diversity_loss_weight > 0.
        # 存储是否具有码本多样性损失
        self.has_codebook_diversity_loss = has_codebook_diversity_loss
        # 存储码本多样性损失的温度参数
        self.codebook_diversity_temperature = codebook_diversity_temperature
        # 存储码本多样性损失的权重
        self.codebook_diversity_loss_weight = codebook_diversity_loss_weight

        # 确保 straight_through 和 rotation_trick 不同时使用
        assert not (straight_through and rotation_trick)
        # 是否使用旋转技巧传递梯度
        self.rotation_trick = rotation_trick

        # 确保 learnable_codebook 与 EMA 更新不兼容
        assert not (ema_update and learnable_codebook), 'learnable codebook not compatible with EMA update'

        # 确保 sync_update_v 在合理范围内，并且 learnable_codebook 必须启用
        assert 0 <= sync_update_v <= 1.
        assert not (sync_update_v > 0. and not learnable_codebook), 'learnable codebook must be turned on'

        # 控制同步更新规则中乐观与悲观更新的参数
        self.sync_update_v = sync_update_v

        # 根据是否使用余弦相似度选择码本类
        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        # 定义 Gumbel 采样函数
        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic = stochastic_sample_codes,
            straight_through = straight_through
        )

        # 如果未提供 sync_codebook，则默认为分布式环境
        if not exists(sync_codebook):
            sync_codebook = is_distributed()

        # 定义码本参数
        codebook_kwargs = dict(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1, # 码本数量
            codebook_size = codebook_size, # 码本大小
            kmeans_init = kmeans_init, # 是否使用 K-Means 初始化
            kmeans_iters = kmeans_iters, # K-Means 聚类的迭代次数
            sync_kmeans = sync_kmeans, # 是否在分布式环境中同步 K-Means
            decay = decay, # EMA 的衰减因子
            eps = eps, # 防止除以零的极小值
            threshold_ema_dead_code = threshold_ema_dead_code, # EMA 死码的阈值
            use_ddp = sync_codebook, # 是否使用分布式数据并行
            learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook, # 是否使码本可学习
            sample_codebook_temp = sample_codebook_temp, # 采样码本的温度参数
            gumbel_sample = gumbel_sample_fn, # Gumbel 采样函数
            ema_update = ema_update, # 是否使用 EMA 更新
            manual_ema_update = manual_ema_update # 是否手动更新 EMA
        )

        # 如果使用仿射参数，则添加相关参数
        if affine_param:
            assert not use_cosine_sim, 'affine param is only compatible with euclidean codebook'
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param = True, # 是否使用仿射参数
                sync_affine_param = sync_affine_param, # 是否在分布式环境中同步仿射参数
                affine_param_batch_decay = affine_param_batch_decay, # 仿射参数的批次衰减因子
                affine_param_codebook_decay = affine_param_codebook_decay, # 仿射参数的码本衰减因子
            )

        # 是否使用余弦相似度
        self.use_cosine_sim = use_cosine_sim
        # 初始化码本
        self._codebook = codebook_class(**codebook_kwargs)

        # 初始化就地优化器
        self.in_place_codebook_optimizer = in_place_codebook_optimizer(self._codebook.parameters()) if exists(in_place_codebook_optimizer) else None
        # 是否手动更新就地优化器
        self.manual_in_place_optimizer_update = manual_in_place_optimizer_update

        # 存储码本大小
        self.codebook_size = codebook_size

        # 是否接受图像特征图
        self.accept_image_fmap = accept_image_fmap
        # 是否将通道维度放在最后
        self.channel_last = channel_last

        # 注册缓冲区
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # for variable lengthed sequences, whether to take care of masking out the padding to 0 (or return the original input)
        # 对于可变长度的序列，是否将填充掩码为 0（或返回原始输入）
        self.return_zeros_for_masked_padding = return_zeros_for_masked_padding

    @property
    def codebook(self):
        """
        获取码本嵌入。

        返回:
            Tensor: 码本嵌入。
        """
        # 获取码本嵌入
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            # 如果为每个注意力头使用独立的码本，则直接返回
            return codebook

        # 否则，重塑码本嵌入
        return rearrange(codebook, '1 ... -> ...')

    @codebook.setter
    def codebook(self, codes):
        """
        设置码本嵌入。

        参数:
            codes (Tensor): 新的码本嵌入。
        """
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, '... -> 1 ...')

        self._codebook.embed.copy_(codes)

    def get_codes_from_indices(self, indices):
        """
        根据索引从码本中获取对应的码字。

        参数:
            indices (Tensor): 输入的索引张量。

        返回:
            Tensor: 从码本中获取的码字张量。
        """
        # 获取码本
        codebook = self.codebook
        # 判断码本是否为多头
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            # 如果不是多头，直接根据索引从码本中获取码字
            codes = codebook[indices]
        else:
            # 如果是多头，则需要处理多头索引
            # 对多头索引进行打包
            indices, unpack_one = pack_one(indices, 'b * h')
            # 重塑索引张量形状
            indices = rearrange(indices, 'b n h -> b h n')

            # 重复索引以匹配码本维度
            indices = repeat(indices, 'b h n -> b h n d', d = codebook.shape[-1])
            # 重复码本以匹配索引的批次大小
            codebook = repeat(codebook, 'h n d -> b h n d', b = indices.shape[0])

            # 根据索引从码本中获取码字
            codes = codebook.gather(2, indices)
            # 重塑码字张量形状
            codes = rearrange(codes, 'b h n d -> b n (h d)')
            # 解包码字张量
            codes = unpack_one(codes, 'b * d')

        if not self.channel_last:
            # 如果不是通道最后，则调整张量形状
            codes = rearrange(codes, 'b ... d -> b d ...')
        # 返回从码本中获取的码字
        return codes

    def get_output_from_indices(self, indices):
        """
        根据索引获取输出张量。

        参数:
            indices (Tensor): 输入的索引张量。

        返回:
            Tensor: 输出张量。
        """
        # 根据索引获取码字
        codes = self.get_codes_from_indices(indices)
        # 通过投影层获取输出
        return self.project_out(codes)

    def update_in_place_optimizer(self):
        """
        更新就地优化器。
        """
        if not exists(self.in_place_codebook_optimizer):
            return # 如果就地优化器不存在，则返回
        # 更新优化器
        self.in_place_codebook_optimizer.step()
        self.in_place_codebook_optimizer.zero_grad()

    def maybe_split_heads_from_input(self, x):
        """
        将输入张量拆分为多头张量。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 拆分后的多头张量。
        """
        if self.heads == 1: 
            return x # 如果只有一个头，则直接返回输入

        # 根据是否多头调整张量形状
        ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
        return rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = self.heads)

    def expire_codes_(self, x):
        """
        过期码本中的死码。

        参数:
            x (Tensor): 输入张量。
        """
        # 对输入张量进行变换
        x = self._codebook.transform_input(x)
        # 拆分输入张量为多头
        x = self.maybe_split_heads_from_input(x)
        # 调用码本的过期码方法
        self._codebook.expire_codes_(x)

    def forward(
        self,
        x,
        indices = None,
        mask = None,
        lens = None,
        sample_codebook_temp = None,
        freeze_codebook = False,
        return_loss_breakdown = False,
        codebook_transform_fn: Callable | None = None
    ):
        """
        前向传播函数，用于量化输入张量。

        参数:
            x (Tensor): 输入张量。
            indices (Optional[Tensor], 可选): 输入的索引张量。默认值为 None。
            mask (Optional[Tensor], 可选): 输入的掩码张量。默认值为 None。
            lens (Optional[Tensor], 可选): 输入的序列长度张量。默认值为 None。
            sample_codebook_temp (Optional[float], 可选): 采样码本的温度参数。默认值为 None。
            freeze_codebook (bool, 可选): 是否冻结码本。默认值为 False。
            return_loss_breakdown (bool, 可选): 是否返回损失分解。默认值为 False。
            codebook_transform_fn (Optional[Callable], 可选): 码本转换函数。默认值为 None。

        返回:
            Tuple[Tensor, Tensor, Tensor]: 返回量化后的张量、嵌入索引和张量的距离。
        """
        # 保存原始输入
        orig_input = x

        # handle masking, either passed in as `mask` or `lens`
        # 处理掩码，可以通过 `mask` 或 `lens` 传入
        
        # 确保 mask 和 lens 不同时存在
        assert not (exists(mask) and exists(lens))

        if exists(lens):
            # 根据序列长度生成掩码
            mask = lens_to_mask(lens, x.shape[1])

        # handle one token given
        # 处理单个 token 的情况

        # 判断是否为单个 token
        only_one = x.ndim == 2

        if only_one:
            # 确保单个 token 不支持掩码
            assert not exists(mask)
            # 重塑张量形状
            x = rearrange(x, 'b d -> b 1 d')

        # 获取形状、设备、注意力头数量、是否多头、码本大小和是否有索引
        shape, device, heads, is_multiheaded, codebook_size, return_loss = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size, exists(indices)
        
        # 判断是否需要转置
        # 如果不是通道最后且不接受图像特征图，则需要转置
        need_transpose = not self.channel_last and not self.accept_image_fmap
        # 判断是否需要进行就地优化
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        # rearrange inputs

        if self.accept_image_fmap:
            # 确保图像特征图不支持掩码
            assert not exists(mask)
            # 获取高度和宽度
            height, width = x.shape[-2:]
            # 重塑张量形状
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            # 转置张量形状
            x = rearrange(x, 'b d n -> b n d')

        # project input
        # 输入投影

        # 通过投影层进行输入投影
        x = self.project_in(x)

        # handle multi-headed separate codebooks
        # 处理多头独立的码本

        # 根据多头情况拆分输入张量
        x = self.maybe_split_heads_from_input(x)

        # l2norm for cosine sim, otherwise identity
        # 对输入张量进行 L2 归一化（用于余弦相似度）或保持不变

        # 对输入张量进行变换
        x = self._codebook.transform_input(x)

        # codebook forward kwargs
        # 码本前向传播参数

        codebook_forward_kwargs = dict(
            sample_codebook_temp = sample_codebook_temp, # 采样码本的温度参数
            mask = mask, # 掩码
            freeze_codebook = freeze_codebook, # 是否冻结码本
            codebook_transform_fn = codebook_transform_fn # 码本转换函数
        )

        # quantize
        # 量化

        # 调用码本进行量化
        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        # losses for loss breakdown
        # 损失

        # 初始化损失
        commit_loss = orthogonal_reg_loss = inplace_optimize_loss = codebook_diversity_loss = self.zero

        # one step in-place update
        # 进行优化

        if should_inplace_optimize and self.training and not freeze_codebook:

            if exists(mask):
                # 计算均方误差损失
                loss = F.mse_loss(quantize, x.detach(), reduction = 'none')

                loss_mask = mask
                if is_multiheaded:
                    # 重复掩码以匹配多头情况
                    loss_mask = repeat(mask, 'b n -> c (b h) n', c = loss.shape[0], h = loss.shape[1] // mask.shape[0])

                    # 计算掩码后的均方误差损失
                    loss = loss[loss_mask].mean()

            else:
                # 计算均方误差损失
                loss = F.mse_loss(quantize, x.detach())

            loss.backward()

            if not self.manual_in_place_optimizer_update:
                self.update_in_place_optimizer()

            inplace_optimize_loss = loss

            # quantize again
            # 重新量化

            # 重新调用码本进行量化
            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.training:
            # determine code to use for commitment loss
            # 确定用于承诺损失的代码
            maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity

            commit_quantize = maybe_detach(quantize)

            if self.rotation_trick:
                # 应用旋转技巧
                quantize = rotate_to(x, quantize)
            else:
                # standard STE to get gradients through VQ layer.
                # 使用标准直通估计（STE）传递梯度
                quantize = x + (quantize - x).detach()

            if self.sync_update_v > 0.:
                quantize = quantize + self.sync_update_v * (quantize - quantize.detach())

        # function for calculating cross entropy loss to distance matrix
        # 计算交叉熵损失到距离矩阵的函数
        # used for (1) naturalspeech2 training residual vq latents to be close to the correct codes and (2) cross-entropy based commitment loss

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = '1 b n l -> b l n' # 单头情况
            elif self.separate_codebook_per_head:
                dist_einops_eq = 'c b n l -> b l n c' # 多头且每个头有独立码本
            else:
                dist_einops_eq = '1 (b h) n l -> b l n h' # 多头但共享码本

            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b = shape[0]), # 重塑距离矩阵
                codes,
                ignore_index = -1
            )

            # 返回交叉熵损失
            return ce_loss

        # if returning cross entropy loss on codes that were passed in
        # 如果需要返回传递进来的编码的交叉熵损失

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        # transform embedding indices
        # 转换嵌入索引

        if is_multiheaded:
            if self.separate_codebook_per_head:
                # 如果每个头有单独的码本，则重新排列形状为 (batch_size, num_tokens, num_heads)
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h = heads)
            else:
                # 否则，重新排列形状为 (batch_size, num_tokens, num_heads)
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h = heads)

        if self.accept_image_fmap:
            # 如果接受图像特征图，则重新排列形状为 (batch_size, height, width, ...)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h = height, w = width)

        if only_one:
            # 如果只有一个嵌入索引，则移除维度
            embed_ind = rearrange(embed_ind, 'b 1 ... -> b ...')

        # aggregate loss

        # 初始化损失为标量张量
        loss = torch.tensor([0.], device = device, requires_grad = self.training)

        if self.training:
            # calculate codebook diversity loss (negative of entropy) if needed
            # 如果在训练模式，计算码本多样性损失（熵的负值）如果需要

            if self.has_codebook_diversity_loss:
                # 计算概率分布，使用温度缩放
                prob = (-distances * self.codebook_diversity_temperature).softmax(dim = -1)
                # 计算平均概率
                avg_prob = reduce(prob, '... n l -> n l', 'mean')
                # 计算多样性损失（熵的负值）
                codebook_diversity_loss = -entropy(avg_prob).mean()
                # 将多样性损失加入总损失
                loss = loss + codebook_diversity_loss * self.codebook_diversity_loss_weight

            # commitment loss

            if self.has_commitment_loss:
                if self.commitment_use_cross_entropy_loss:
                    # 使用交叉熵损失计算承诺损失
                    if exists(mask):
                        ce_loss_mask = mask
                        if is_multiheaded:
                            # 如果是多头，并且有掩码，则重复掩码以匹配头的数量
                            ce_loss_mask = repeat(ce_loss_mask, 'b n -> b n h', h = heads)
                        # 使用掩码填充嵌入索引
                        embed_ind.masked_fill_(~ce_loss_mask, -1)
                    # 计算承诺损失
                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if exists(mask):
                        # with variable lengthed sequences
                        # 如果存在掩码且使用均方误差损失
                        # 计算均方误差损失，不进行归约
                        commit_loss = F.mse_loss(commit_quantize, x, reduction = 'none')
                        # 应用掩码
                        loss_mask = mask
                        if is_multiheaded:
                            # 如果是多头，则重复掩码以匹配损失形状
                            loss_mask = repeat(loss_mask, 'b n -> c (b h) n', c = commit_loss.shape[0], h = commit_loss.shape[1] // mask.shape[0])
                        # 计算均方误差损失的平均值
                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        # 如果没有掩码，则直接计算均方误差损失
                        commit_loss = F.mse_loss(commit_quantize, x)
                # 将承诺损失加入总损失
                loss = loss + commit_loss * self.commitment_weight
            # 如果需要计算码本正交损失
            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                # only calculate orthogonal loss for the activated codes for this batch
                # 仅对当前批次激活的码本计算正交损失
                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), 'orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet'
                    # 获取唯一的嵌入索引
                    unique_code_ids = torch.unique(embed_ind)
                    # 提取对应的码本嵌入向量
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]
                # 如果指定了最大码本数量，并且当前数量超过限制，则随机选择一部分码本
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device = device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[:, rand_ids]
                # 计算正交损失
                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                # 将正交损失加入总损失
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # handle multi-headed quantized embeddings
        # 处理多头量化嵌入

        if is_multiheaded:
            if self.separate_codebook_per_head:
                # 如果每个头有单独的码本，则重新排列量化嵌入的形状为 (batch_size, num_tokens, head_dim * num_heads)
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h = heads)
            else:
                # 否则，重新排列量化嵌入的形状为 (batch_size, num_tokens, head_dim * num_heads)
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h = heads)

        # project out
        # 投影输出层
        quantize = self.project_out(quantize)

        # rearrange quantized embeddings
        # 重新排列量化嵌入的形状
        if need_transpose:
            # 如果需要转置，则将形状从 (batch_size, num_tokens, dim) 转换为 (batch_size, dim, num_tokens)
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            # 如果接受图像特征图，则将形状从 (batch_size, height * width, channels) 转换为 (batch_size, channels, height, width)
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h = height, w = width)

        if only_one:
            # 如果只有一个量化嵌入，则移除维度，形状从 (batch_size, 1, dim) 转换为 (batch_size, dim)
            quantize = rearrange(quantize, 'b 1 d -> b d')

        # if masking, only return quantized for where mask has True
        # 如果存在掩码，则只返回掩码为 True 的位置的量化嵌入

        if exists(mask):
            # 定义掩码为 False 时的输出值
            masked_out_value = orig_input

            if self.return_zeros_for_masked_padding:
                # 如果需要返回零，则将掩码为 False 的位置设置为零张量
                masked_out_value = torch.zeros_like(orig_input)
            # 使用掩码选择量化嵌入或原始输入
            quantize = einx.where(
                'b n, b n d, b n d -> b n d', # 条件、选择1、选择2 -> 输出形状
                mask,
                quantize,
                masked_out_value
            )
            # 使用掩码选择嵌入索引或填充为 -1
            embed_ind = einx.where(
                'b n, b n ..., -> b n ...',
                mask,
                embed_ind,
                -1
            )
        # 如果不需要返回损失分解，则返回量化嵌入、嵌入索引和总损失
        if not return_loss_breakdown:
            return quantize, embed_ind, loss
        # 创建损失分解对象，包含各项具体损失
        loss_breakdown = LossBreakdown(commit_loss, codebook_diversity_loss, orthogonal_reg_loss, inplace_optimize_loss)
        # 返回量化嵌入、嵌入索引、总损失和损失分解
        return quantize, embed_ind, loss, loss_breakdown
