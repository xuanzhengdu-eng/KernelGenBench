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

@label("moe_lora_align_block_size")
@parametrize("config", [
    (16, 2, 8, 4, 2),
    (32, 2, 8, 8, 4),
    (64, 4, 16, 4, 2),
    (128, 2, 16, 8, 4),
    (128, 4, 16, 4, 2),
    (32, 4, 8, 4, 2),
    (256, 4, 16, 8, 4),
    (256, 2, 16, 4, 2),
    (128, 2, 32, 8, 4),
    (64, 2, 8, 4, 2),
])
def test_accuracy_moe_lora_align_block_size(config):
    num_tokens, topk, num_experts, block_size, max_loras = config
    max_num_tokens_padded = num_tokens * topk + num_experts * block_size
    max_num_m_blocks = max_num_tokens_padded // block_size
    token_count = num_tokens * topk

    def make_topk_ids():
        return torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)

    lora_ids = torch.arange(max_loras, device=device, dtype=torch.int32)
    mapping = torch.arange(num_tokens, device=device, dtype=torch.int32) % max_loras
    enabled = torch.ones_like(lora_ids)

    def make_output_buffers():
        return (
            torch.full((max_loras * max_num_tokens_padded,), token_count, device=device, dtype=torch.int32),
            torch.full((max_loras * max_num_m_blocks,), -1, device=device, dtype=torch.int32),
            torch.zeros_like(lora_ids),
        )

    def run(operation, ids, outputs):
        operation(
            ids, mapping, num_experts, block_size,
            max_loras, max_num_tokens_padded, max_num_m_blocks,
            *outputs, enabled, lora_ids, None,
        )

    baseline = kernelgenbench.baseline.moe_lora_align_block_size
    candidate = kernelgenbench.triton.moe_lora_align_block_size
    ids = make_topk_ids()
    ref_output = make_output_buffers()
    act_output = make_output_buffers()
    run(baseline, ids, ref_output)
    run(candidate, ids, act_output)
    ref_sorted, ref_experts, ref_ntp = ref_output
    act_sorted, act_experts, act_ntp = act_output

    assert (ref_ntp > 0).all(), "Every test adapter must route at least one token"
    assert torch.equal(act_ntp, ref_ntp), "Padded token counts do not match"
    assert torch.equal(act_experts, ref_experts), "Expert ids do not match"
    assert ((act_sorted >= 0) & (act_sorted <= token_count)).all(), "Invalid token id"

    # The native large-batch path assigns positions with atomic increments, so
    # the order is unspecified within an expert. Keep adapter and expert
    # boundaries, token multiplicity, and the padding sentinel in the comparison.
    expert_groups = ref_experts.reshape(max_loras, -1).repeat_interleave(block_size, dim=1)
    expert_keys = (expert_groups.to(torch.int64) + 1) * (token_count + 1)
    ref_keys = expert_keys + ref_sorted.reshape(max_loras, -1)
    act_keys = expert_keys + act_sorted.reshape(max_loras, -1)
    assert torch.equal(act_keys.sort(dim=1).values, ref_keys.sort(dim=1).values), "Token routing does not match"

    if num_tokens < 128 or num_experts < 16:
        return None

    ids_bench = make_topk_ids()
    ref_bench = make_output_buffers()
    act_bench = make_output_buffers()
    ms_baseline = triton.testing.do_bench(lambda: run(baseline, ids_bench, ref_bench), warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(lambda: run(candidate, ids_bench, act_bench), warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
