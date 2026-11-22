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

def make_packed_cache(num_blocks, block_size, head_dim):
    # vLLM 0.13 uses one FP32 scale per 128 bytes. Each cache block
    # contains all key bytes followed by the FP32 scales for that block.
    keys = torch.randint(0, 256, (num_blocks, block_size, head_dim),
                         device=device, dtype=torch.uint8)
    scales = torch.rand(num_blocks, block_size, head_dim // 128, device=device)
    scale_bytes = scales.view(torch.uint8)
    cache = torch.cat((keys.flatten(1), scale_bytes.flatten(1)), dim=1)
    return cache.view(num_blocks, block_size, -1), keys, scale_bytes


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
    num_blocks, block_size, num_heads, head_size, batch_size = config
    head_dim = num_heads * head_size
    kv_cache, keys, scales = make_packed_cache(num_blocks, block_size, head_dim)
    blocks_per_seq = 2
    total_tokens = batch_size * blocks_per_seq * block_size

    block_table = torch.randint(0, num_blocks, (batch_size, blocks_per_seq),
                                device=device, dtype=torch.int32)
    cu_seq = torch.arange(0, total_tokens + 1,
                          blocks_per_seq * block_size,
                          device=device, dtype=torch.int32)

    selected_blocks = block_table.flatten().long()
    expected_k = keys[selected_blocks].reshape(total_tokens, head_dim)
    expected_s = scales[selected_blocks].reshape(total_tokens, -1)
    ref_k, act_k = torch.empty_like(expected_k), torch.empty_like(expected_k)
    ref_s, act_s = torch.empty_like(expected_s), torch.empty_like(expected_s)

    kernelgenbench.baseline.cp_gather_indexer_k_quant_cache(
        kv_cache, ref_k, ref_s, block_table, cu_seq)
    kernelgenbench.triton.cp_gather_indexer_k_quant_cache(
        kv_cache, act_k, act_s, block_table, cu_seq)

    torch.testing.assert_close(ref_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(ref_s, expected_s, rtol=0, atol=0)
    torch.testing.assert_close(act_k, ref_k, rtol=0, atol=0)
    torch.testing.assert_close(act_s, ref_s, rtol=0, atol=0)

    if num_blocks < 16 or batch_size < 4:
        return None

    kv_cache_bench, _, _ = make_packed_cache(num_blocks, block_size, head_dim)
    k_baseline, k_triton = torch.empty_like(ref_k), torch.empty_like(ref_k)
    s_baseline, s_triton = torch.empty_like(ref_s), torch.empty_like(ref_s)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.cp_gather_indexer_k_quant_cache(
            kv_cache_bench, k_baseline, s_baseline, block_table, cu_seq),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.cp_gather_indexer_k_quant_cache(
            kv_cache_bench, k_triton, s_triton, block_table, cu_seq),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
