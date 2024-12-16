"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast

from einops import rearrange, pack, unpack


# helper functions

def exists(v):
    """
    检查一个值是否存在（不为 None）。

    参数:
        v: 需要检查的值。

    返回:
        bool: 如果 v 不为 None，则返回 True；否则返回 False。
    """
    return v is not None


def default(*args):
    """
    返回第一个存在（不为 None）的值。

    参数:
        *args: 需要检查的可选值。

    返回:
        Any: 返回第一个存在（不为 None）的值。如果所有值都不存在，则返回 None。
    """
    for arg in args:
        if exists(arg):
            return arg
    return None


def maybe(fn):
    """
    装饰器，用于对函数进行包装，使其在输入为 None 时返回输入本身。

    参数:
        fn (callable): 需要包装的函数。

    返回:
        callable: 包装后的函数。
    """
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner


def pack_one(t, pattern):
    """
    将单个张量 t 按照指定的 pattern 打包。

    参数:
        t (Tensor): 需要打包的张量。
        pattern (str): 打包的模式，例如 'b *' 表示批次维度和其他维度。

    返回:
        Tensor: 打包后的张量。
    """
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    """
    将打包后的张量 t 按照指定的 pattern 和打包参数 ps 进行解包，并返回第一个解包后的张量。

    参数:
        t (Tensor): 需要解包的张量。
        ps: 打包参数。
        pattern (str): 解包的模式。

    返回:
        Tensor: 解包后的第一个张量。
    """
    return unpack(t, ps, pattern)[0]


# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    """
    使用直通梯度估计进行四舍五入操作。

    参数:
        z (Tensor): 输入张量。

    返回:
        Tensor: 四舍五入后的张量，梯度通过直通方式传递。
    """
    zhat = z.round()
    return z + (zhat - z).detach()


# main class

class FSQ(Module):
    """
    FSQ (Fixed-size Quantizer) 类，用于固定尺寸的量化器。

    参数:
        levels (List[int]): 每个量化级别的数量。
        dim (int | None, 可选): 输入数据的维度。如果未提供，则默认为 levels 的长度乘以 num_codebooks。
        num_codebooks (int, 可选): 码本的数量。默认值为 1。
        keep_num_codebooks_dim (bool | None, 可选): 是否保留码本数量的维度。如果 num_codebooks 大于 1，则默认为 True。默认值为 None。
        scale (float | None, 可选): 缩放因子。默认值为 None。
        allowed_dtypes (Tuple[torch.dtype, ...], 可选): 允许的数据类型。默认值为 (torch.float32, torch.float64)。
        channel_first (bool, 可选): 是否为通道优先。默认值为 False。
        projection_has_bias (bool, 可选): 线性投影层是否包含偏置。默认值为 True。
        return_indices (bool, 可选): 是否返回量化索引。默认值为 True。
        force_quantization_f32 (bool, 可选): 是否强制量化使用 32 位浮点数。默认值为 True。
    """
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices = True,
        force_quantization_f32 = True
    ):
        super().__init__()
        # 将 levels 转换为整数张量
        _levels = torch.tensor(levels, dtype=int32)
        # 注册 levels 张量为缓冲区，不持久化
        self.register_buffer("_levels", _levels, persistent = False)

        # 计算码本的基础索引（累积乘积）
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
         # 注册 basis 张量为缓冲区，不持久化
        self.register_buffer("_basis", _basis, persistent = False)

        # 存储缩放因子
        self.scale = scale

        # 码本的维度为 levels 的长度
        codebook_dim = len(levels)
        # 存储码本维度
        self.codebook_dim = codebook_dim

        # 有效的码本维度为码本维度乘以码本数量
        effective_codebook_dim = codebook_dim * num_codebooks
        # 存储码本数量
        self.num_codebooks = num_codebooks
        # 存储有效码本维度
        self.effective_codebook_dim = effective_codebook_dim

        # 如果未提供 keep_num_codebooks_dim，则默认为 num_codebooks > 1
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        # 确保如果码本数量大于 1，则保留码本数量的维度
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        # 存储是否保留码本数量的维度
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # 如果未提供 dim，则默认为 levels 的长度乘以码本数量
        self.dim = default(dim, len(_levels) * num_codebooks)

        # 存储是否为通道优先
        self.channel_first = channel_first

        # 判断是否需要进行线性投影
        has_projections = self.dim != effective_codebook_dim
        # 如果需要，进行线性投影；否则，使用恒等变换
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        # 如果需要，进行线性投影；否则，使用恒等变换
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        # 存储是否需要进行线性投影
        self.has_projections = has_projections

        # 存储是否返回量化索引
        self.return_indices = return_indices
        if return_indices:
            # 计算码本的大小
            self.codebook_size = self._levels.prod().item()
            # 生成隐式码本
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            # 注册隐式码本为缓冲区，不持久化
            self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        # 存储允许的数据类型
        self.allowed_dtypes = allowed_dtypes
        # 存储是否强制量化使用 32 位浮点数
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        """
        对输入张量 z 进行边界限制。

        参数:
            z (Tensor): 输入张量，形状为 (..., d)。
            eps (float, 可选): 边界限制的精度。默认值为 1e-3。

        返回:
            Tensor: 边界限制后的张量。
        """
        # 计算边界的一半长度，_levels 是量化级别
        half_l = (self._levels - 1) * (1 + eps) / 2
        # 如果量化级别是偶数，则偏移量为 0.5；否则为 0.0
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        # 计算偏移量的反正切双曲函数
        shift = (offset / half_l).atanh()
        # 应用边界限制，返回限制后的张量
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        """
        对输入张量 z 进行量化。

        参数:
            z (Tensor): 输入张量。

        返回:
            Tensor: 量化后的张量，形状与 z 相同。
        """
        # 对边界限制后的张量进行四舍五入（使用直通梯度估计）
        quantized = round_ste(self.bound(z))
        # 计算量化级别的一半宽度，用于归一化到 [-1, 1]
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        # 返回归一化后的量化张量
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized):
        """
        将归一化后的量化张量 zhat_normalized 进行缩放和平移。

        参数:
            zhat_normalized (Tensor): 归一化后的量化张量。

        返回:
            Tensor: 缩放和平移后的张量。
        """
        # 计算量化级别的一半宽度
        half_width = self._levels // 2
        # 返回缩放和平移后的张量
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        """
        将量化张量 zhat 进行逆缩放和平移。

        参数:
            zhat (Tensor): 量化张量。

        返回:
            Tensor: 逆缩放和平移后的归一化量化张量。
        """
        # 计算量化级别的一半宽度
        half_width = self._levels // 2
        # 返回逆缩放和平移后的归一化量化张量
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        """
        将索引转换为码本中的代码。

        参数:
            indices (Tensor): 索引张量。

        返回:
            Tensor: 转换后的代码张量。
        """
        # 将索引转换为每个级别的索引
        level_indices = self.indices_to_level_indices(indices)
        # 对每个级别的索引进行逆缩放和平移
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        """
        将码本中的代码转换为索引。

        参数:
            zhat (Tensor): 代码张量。

        返回:
            Tensor: 转换后的索引张量。
        """
        # 确保代码张量的最后一个维度与码本维度匹配
        assert zhat.shape[-1] == self.codebook_dim
        # 对代码进行缩放和平移
        zhat = self._scale_and_shift(zhat)
        # 将缩放和平移后的代码转换为索引
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        """
        将索引转换为每个级别的索引，用于因子化嵌入的 transformer。

        参数:
            indices (Tensor): 索引张量。

        返回:
            Tensor: 每个级别的索引张量。
        """
        # 重塑索引张量形状
        indices = rearrange(indices, '... -> ... 1')
        # 计算每个级别的非中心化代码
        codes_non_centered = (indices // self._basis) % self._levels
        # 返回每个级别的索引
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        """
        `codes_to_indices` 的逆操作，将索引转换为码本中的代码。

        参数:
            indices (Tensor): 输入的索引张量。

        返回:
            Tensor: 转换后的代码张量。
        """
        # 确保索引张量存在
        assert exists(indices)
        # 判断是否为图像或视频数据，图像或视频数据通常具有至少 3 个维度（批次、通道、高度、宽度）
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        # 将索引转换为码本中的代码
        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            # 如果保留码本数量的维度，则重塑代码张量形状
            codes = rearrange(codes, '... c d -> ... (c d)')
        # 通过线性投影层将代码转换为输出维度
        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            # 如果是图像或视频数据，或者数据为通道优先，则重塑张量形状，将通道维度移动到最后一个维度
            codes = rearrange(codes, 'b ... d -> b d ...')
        
        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        """
        前向传播方法，对输入张量 z 进行量化。

        参数:
            z (Tensor): 输入张量。

        返回:
            Tuple[Tensor, Optional[Tensor]]: 返回量化后的输出张量和可选的量化索引。
        """
        # 判断是否为图像或视频数据，图像或视频数据通常具有至少 4 个维度（批次、通道、高度、宽度）
        is_img_or_video = z.ndim >= 4
        # 判断是否需要将通道维度移动到最后一个维度
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)
        # 将图像或视频数据标准化为 (批次, 序列, 维度)
        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d') # 重塑张量形状，将通道维度移动到最后一个维度
            z, ps = pack_one(z, 'b * d') # 打包张量，保存打包参数
        # 确保输入张量的最后一个维度与模型维度匹配
        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        # 通过线性投影层将输入张量投影到有效码本维度
        z = self.project_in(z)

        # 重塑张量形状，将有效码本维度拆分为多个码本维度
        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # whether to force quantization step to be full precision or not
        # 是否强制量化步骤为全精度
        force_f32 = self.force_quantization_f32
        # 如果强制为全精度，则在 CUDA 上禁用自动混合精度（AMP）；否则，使用 nullcontext
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype # 保存原始数据类型

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float() # 如果强制为全精度且原始数据类型不在允许的数据类型中，则将张量转换为浮点数类型
            
            # 对张量进行量化
            codes = self.quantize(z)

            # returning indices could be optional
            # 返回索引是可选的
            indices = None

            if self.return_indices:
                # 将量化后的代码转换为索引
                indices = self.codes_to_indices(codes)

            # 重塑代码张量形状，将多个码本维度合并
            codes = rearrange(codes, 'b n c d -> b n (c d)')

            # 将代码张量的数据类型转换回原始数据类型
            codes = codes.type(orig_dtype)

        # project out
        # 通过线性投影层将代码投影回输出维度
        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # 重建图像或视频数据的维度
        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d') # 解包张量，恢复打包前的形状
            out = rearrange(out, 'b ... d -> b d ...') # 重塑张量形状，将通道维度移动回第一个维度

            # 解包索引张量（如果存在），恢复打包前的形状
            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            # 如果不保留码本数量的维度且返回索引，则重塑索引张量形状，去除多余的维度
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices
        # 返回量化后的输出和索引
        return out, indices
