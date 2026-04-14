import math

import torch
import torch.nn.functional as F
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.srt.utils.custom_op import register_custom_op

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _group_norm_silu_contiguous_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    channels,
    spatial_size,
    channels_per_group,
    group_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    group_base = batch_id * channels * spatial_size + group_id * group_size
    offsets = tl.arange(0, BLOCK_SIZE)

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for off in range(0, group_size, BLOCK_SIZE):
        idx = off + offsets
        mask = idx < group_size
        x = tl.load(input_ptr + group_base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    inv_group = 1.0 / group_size
    mean = sum_val * inv_group
    var = sum_sq * inv_group - mean * mean
    rstd = tl.rsqrt(var + eps)

    weight_group_offset = group_id * channels_per_group
    for off in range(0, group_size, BLOCK_SIZE):
        idx = off + offsets
        mask = idx < group_size
        x = tl.load(input_ptr + group_base + idx, mask=mask, other=0.0).to(tl.float32)
        channel_offsets = weight_group_offset + idx // spatial_size
        weight = tl.load(weight_ptr + channel_offsets, mask=mask, other=1.0).to(
            tl.float32
        )
        bias = tl.load(bias_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * weight + bias
        y = y * tl.sigmoid(y)
        tl.store(output_ptr + group_base + idx, y, mask=mask)


def _group_norm_silu_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    return F.silu(F.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps))


def _can_use_triton_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
) -> bool:
    return (
        x.is_cuda
        and not x.requires_grad
        and x.dtype in _SUPPORTED_DTYPES
        and x.ndim in (2, 3, 4, 5)
        and x.shape[1] % num_groups == 0
        and weight.is_cuda
        and bias.is_cuda
        and weight.dtype == x.dtype
        and bias.dtype == x.dtype
        and weight.ndim == 1
        and bias.ndim == 1
        and weight.shape == bias.shape == (x.shape[1],)
    )


@register_custom_op(op_name="triton_group_norm_silu_cuda", out_shape="x")
def _triton_group_norm_silu_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    if not _can_use_triton_group_norm_silu(x, weight, bias, num_groups):
        return _group_norm_silu_native(x, weight, bias, num_groups, eps)

    x_contiguous = x.contiguous()
    batch_size, channels = x_contiguous.shape[:2]
    spatial_size = math.prod(x_contiguous.shape[2:]) if x_contiguous.ndim > 2 else 1
    channels_per_group = channels // num_groups
    group_size = channels_per_group * spatial_size

    x_flat = x_contiguous.reshape(batch_size, channels, spatial_size, 1)
    y_flat = torch.empty_like(x_flat)

    block_size = min(4096, triton.next_power_of_2(max(1, min(group_size, 4096))))
    grid = (num_groups, batch_size)

    with torch.cuda.device(x.device):
        _group_norm_silu_contiguous_kernel[grid](
            x_flat,
            weight,
            bias,
            y_flat,
            channels,
            spatial_size,
            channels_per_group,
            group_size,
            eps,
            BLOCK_SIZE=block_size,
        )

    return y_flat.reshape_as(x_contiguous)


def triton_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    return _triton_group_norm_silu_cuda(x, weight, bias, num_groups, eps)


__all__ = ["triton_group_norm_silu"]
