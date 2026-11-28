# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton


def _indexer_layout(config):
    """Validate the byte layout used by vLLM v0.13's gather kernel.

    The primitive gathers a single indexer key vector, not multiple heads.
    The old head count is retained as the number of independent invocations.
    """
    num_blocks, block_size, num_heads, head_dim, batch_size = config
    if min(config) <= 0 or block_size % 16 or head_dim % 16:
        raise ValueError("positive block sizes and 16-byte-aligned heads required")
    # One scale is written per 128-byte grid-y tile. A shorter head has one
    # scale; larger heads must have exactly one scale per full 128-byte tile.
    if head_dim < 128:
        quant_block_size = head_dim
    elif head_dim % 128 == 0:
        quant_block_size = 128
    else:
        raise ValueError("heads >= 128 bytes must be divisible by 128")
    scale_bytes = 4 * (head_dim // quant_block_size)
    cache_stride = head_dim + scale_bytes
    block_stride = block_size * cache_stride
    if block_stride % 16:
        raise ValueError("cache block starts must be 16-byte aligned")
    blocks_per_seq = 2
    seq_len = blocks_per_seq * block_size
    total_tokens = batch_size * seq_len
    # C++ uses int for head_dim * 4 and ceil-divides token counts in int.
    if (total_tokens > 2**31 - 32 or num_blocks - 1 > 2**31 - 1
            or head_dim > (2**31 - 1) // 4 or head_dim > 128 * 65535):
        raise ValueError("indexer dimensions exceed native int32 or CUDA grid limits")
    return num_blocks, block_size, num_heads, head_dim, batch_size, quant_block_size, scale_bytes, cache_stride, total_tokens


def _make_indexer_cache(num_heads, num_blocks, block_size, head_dim, scale_bytes, cache_stride):
    # vLLM stores all K bytes in a cache block first, then all FP32 scale
    # bytes, not interleaved per token. Each leading slice is an independent
    # contiguous [num_blocks, block_size, cache_stride] single-head cache.
    cache = torch.empty((num_heads, num_blocks, block_size, cache_stride),
                        device=device, dtype=torch.uint8)
    flat = cache.view(num_heads, num_blocks, -1)
    k = torch.randn((num_heads, num_blocks, block_size, head_dim),
                    device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    flat[:, :, :block_size * head_dim] = k.view(torch.uint8).reshape(num_heads, num_blocks, -1)
    scales = torch.rand((num_heads, num_blocks, block_size, scale_bytes // 4),
                        device=device, dtype=torch.float32) + 0.01
    flat[:, :, block_size * head_dim:] = scales.view(torch.uint8).reshape(num_heads, num_blocks, -1)
    return cache


def _check_indexer_buffers(cache, k, scale, block_table, cu_seq, layout):
    num_blocks, block_size, num_heads, head_dim, batch_size, _, scale_bytes, cache_stride, total_tokens = layout
    if (cache.shape != (num_heads, num_blocks, block_size, cache_stride)
            or k.shape != (num_heads, total_tokens, head_dim)
            or scale.shape != (num_heads, total_tokens, scale_bytes)
            or block_table.shape != (batch_size, 2)
            or cu_seq.shape != (batch_size + 1,)):
        raise ValueError("indexer fixture shape does not match the kernel contract")
    if (cache.dtype != torch.uint8 or k.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8
            or block_table.dtype != torch.int32 or cu_seq.dtype != torch.int32):
        raise ValueError("indexer storage must use uint8 cache/scale, FP8 keys and int32 indices")
    if not all(t.is_contiguous() for t in (cache, k, scale, block_table, cu_seq)):
        raise ValueError("indexer buffers and index arrays must be contiguous")
    if not all(t.device == cache.device for t in (k, scale, block_table, cu_seq)):
        raise ValueError("indexer buffers and index arrays must share a device")
    if (cache.data_ptr() % 16 or k.data_ptr() % 16 or scale.data_ptr() % 4
            or block_table.data_ptr() % 4 or cu_seq.data_ptr() % 4):
        raise ValueError("indexer float4/float/int32 accesses require aligned storage")


@label("cp_gather_indexer_k_quant_cache")
@parametrize("config", [
    (4, 16, 4, 64, 2),
    (8, 16, 8, 128, 2),
    (8, 32, 4, 64, 4),
    (16, 16, 8, 64, 2),
    (16, 16, 4, 64, 4),
    (16, 32, 4, 128, 4),
    (32, 16, 4, 64, 4),
    (32, 32, 8, 128, 4),
    (32, 64, 4, 128, 4),
    (64, 32, 8, 128, 4),
    (64, 64, 8, 256, 4),
    (64, 64, 8, 256, 8),
    (128, 64, 8, 256, 8),
])
def test_accuracy_cp_gather_indexer_k_quant_cache(config):
    layout = _indexer_layout(config)
    num_blocks, block_size, num_heads, head_dim, batch_size, _, scale_bytes, cache_stride, total_tokens = layout
    kv_cache = _make_indexer_cache(num_heads, num_blocks, block_size, head_dim, scale_bytes, cache_stride)
    # Each batch has exactly two full blocks. Construction guarantees that
    # every token has a batch, and that all table entries address this cache.
    # Deliberately permute physical blocks so a candidate that ignores the
    # block table cannot pass by treating logical and physical IDs as equal.
    block_table = ((torch.arange(batch_size * 2, device=device, dtype=torch.int32) * 3 + 1)
                   .reshape(batch_size, 2).remainder(num_blocks))
    cu_seq = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * (2 * block_size)
    # Allocate by bytes then view as FP8: exact byte comparison avoids any
    # unsupported FP8 arithmetic while retaining the upstream output dtype.
    ref_k = torch.zeros((num_heads, total_tokens, head_dim), device=device,
                        dtype=torch.uint8).view(torch.float8_e4m3fn)
    act_k = torch.zeros((num_heads, total_tokens, head_dim), device=device,
                        dtype=torch.uint8).view(torch.float8_e4m3fn)
    ref_s = torch.zeros((num_heads, total_tokens, scale_bytes), device=device, dtype=torch.uint8)
    act_s = torch.zeros_like(ref_s)
    _check_indexer_buffers(kv_cache, ref_k, ref_s, block_table, cu_seq, layout)
    _check_indexer_buffers(kv_cache, act_k, act_s, block_table, cu_seq, layout)

    def gather_all(op, cache, out_k, out_s):
        for head in range(num_heads):
            op(cache[head], out_k[head], out_s[head], block_table, cu_seq)

    gather_all(kernelgenbench.baseline.cp_gather_indexer_k_quant_cache, kv_cache, ref_k, ref_s)
    gather_all(kernelgenbench.triton.cp_gather_indexer_k_quant_cache, kv_cache, act_k, act_s)
    ref_k_bytes = ref_k.view(torch.uint8)
    act_k_bytes = act_k.view(torch.uint8)
    # Independently derive the expected bytes from the block table and the
    # documented non-interleaved cache layout. Ref-vs-ref equality alone
    # would not detect an identically wrong result on both sides.
    token = torch.arange(total_tokens, device=device, dtype=torch.long)
    seq_pos = token.remainder(2 * block_size)
    batch = token // (2 * block_size)
    block_id = block_table[batch, seq_pos // block_size].to(torch.long)
    offset = seq_pos.remainder(block_size)
    flat_cache = kv_cache.view(num_heads, num_blocks, -1)
    key_cache = flat_cache[:, :, :block_size * head_dim].view(num_heads, num_blocks, block_size, head_dim)
    scale_cache = flat_cache[:, :, block_size * head_dim:].view(num_heads, num_blocks, block_size, scale_bytes)
    expected_k = key_cache[:, block_id, offset, :]
    expected_s = scale_cache[:, block_id, offset, :]
    assert torch.equal(ref_k_bytes, expected_k), 'reference K bytes differ from the independent block-table oracle'
    assert torch.equal(ref_s, expected_s), 'reference scale bytes differ from the independent block-table oracle'
    assert torch.equal(act_k_bytes, expected_k), 'candidate K bytes differ from the independent block-table oracle'
    assert torch.equal(act_s, expected_s), 'candidate scale bytes differ from the independent block-table oracle'

    # Repeat all independent single-head invocations in each timing sample.
    kv_cache_bench = _make_indexer_cache(num_heads, num_blocks, block_size, head_dim, scale_bytes, cache_stride)
    k_baseline = torch.zeros((num_heads, total_tokens, head_dim), device=device,
                             dtype=torch.uint8).view(torch.float8_e4m3fn)
    s_baseline = torch.zeros_like(ref_s)
    k_triton = torch.zeros((num_heads, total_tokens, head_dim), device=device,
                           dtype=torch.uint8).view(torch.float8_e4m3fn)
    s_triton = torch.zeros_like(act_s)
    _check_indexer_buffers(kv_cache_bench, k_baseline, s_baseline, block_table, cu_seq, layout)
    _check_indexer_buffers(kv_cache_bench, k_triton, s_triton, block_table, cu_seq, layout)
    ms_baseline = triton.testing.do_bench(
        lambda: gather_all(kernelgenbench.baseline.cp_gather_indexer_k_quant_cache,
                           kv_cache_bench, k_baseline, s_baseline),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: gather_all(kernelgenbench.triton.cp_gather_indexer_k_quant_cache,
                           kv_cache_bench, k_triton, s_triton),
        warmup=25, rep=100)
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
