"""Explicit opt-in adapter for a private, separately built SM90 AllSpark op.

The H20 extension is not a replacement for vLLM's global dispatcher.  It is
loaded only when KERNELGENBENCH_H20_ALLSPARK_LIBRARY names a built library.
"""
from functools import lru_cache
import os
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _load_library(path):
    torch.ops.load_library(path)
    return torch.ops.kg_allspark_h20


def h20_allspark_ops(tensor):
    """Return the opted-in private op namespace, or None for stock vLLM."""
    path = os.environ.get('KERNELGENBENCH_H20_ALLSPARK_LIBRARY')
    if not path:
        return None
    if not tensor.is_cuda or torch.cuda.get_device_capability(tensor.device) != (9, 0):
        raise RuntimeError('The H20 AllSpark extension requires an SM90 CUDA tensor')
    if torch.__version__.split('+')[0] != '2.9.0' or torch.version.cuda != '12.9':
        raise RuntimeError('The H20 AllSpark extension was built for PyTorch 2.9.0/CUDA 12.9')
    return _load_library(str(Path(path).expanduser().resolve(strict=True)))


def allspark_repack_weight(qweight, scale, zero_point=None, has_zp=False):
    """Repack with the private SM90 op, or use stock vLLM when not opted in."""
    ops = h20_allspark_ops(qweight)
    if ops is None:
        from vllm import _custom_ops
        return _custom_ops.allspark_repack_weight(qweight, scale, zero_point, has_zp)
    if qweight.ndim != 2 or qweight.dtype != torch.uint8 or not qweight.is_contiguous():
        raise ValueError('qweight must be a contiguous uint8 [K, N] tensor')
    k, n = qweight.shape
    if k <= 0 or n <= 0 or k % 16 or n % 16:
        raise ValueError('AllSpark repack requires positive K and N divisible by 16')
    if (scale.shape != (1, n) or scale.dtype not in (torch.float16, torch.bfloat16)
            or scale.device != qweight.device or not scale.is_contiguous()):
        raise ValueError('scale must be contiguous [1, N] FP16/BF16 on the qweight device')
    if has_zp and (zero_point is None or zero_point.shape != scale.shape
                   or zero_point.dtype != scale.dtype or zero_point.device != scale.device
                   or not zero_point.is_contiguous()):
        raise ValueError('zero_point must match scale when has_zp is true')
    padded_n = ((n + 31) // 32) * 32
    q_reordered = torch.empty((padded_n, k), dtype=qweight.dtype, device=qweight.device)
    s_reordered = torch.empty((1, padded_n), dtype=scale.dtype, device=scale.device)
    z_reordered = torch.empty_like(s_reordered) if has_zp else None
    ops.rearrange_kn_weight_as_n32k16_order(qweight, scale, zero_point, has_zp,
                                             q_reordered, s_reordered, z_reordered,
                                             k, n, padded_n)
    return q_reordered, s_reordered, z_reordered
