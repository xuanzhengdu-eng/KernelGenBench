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
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

def _gptq_layout(q_weight_bit):
    """Validate the fixture dimensions against vLLM's packed-row kernel."""
    (rows, config_cols), bit = q_weight_bit
    if bit not in (4, 8) or rows <= 0 or rows % bit:
        raise ValueError("GPTQ packed rows must be a positive multiple of bit")
    if config_cols <= 0 or config_cols % 32:
        raise ValueError("fixture column parameter must be a positive multiple of 32")
    # Retain the original fixture's actual q_weight widths. The kernel packs
    # along dimension 0, not dimension 1; q_weight is [packed_K, N].
    width = config_cols // 32 * bit
    if width % 2:
        raise ValueError("GPTQ shuffle requires an even N for uint64_t accesses")
    unpacked_rows = rows * (32 // bit)
    if unpacked_rows > 2**31 - 1 or rows * width > 2**31 - 1:
        raise ValueError("GPTQ kernel dimensions and int32 offset products must fit in int32")
    return rows, width, unpacked_rows


def _check_gptq_buffers(q_weight, q_perm, rows, width, unpacked_rows):
    if (q_weight.shape != (rows, width) or q_weight.dtype != torch.int32
            or q_perm.shape != (unpacked_rows,) or q_perm.dtype != torch.int32):
        raise ValueError("GPTQ packed weight or permutation does not match kernel dimensions")
    if not q_weight.is_contiguous() or not q_perm.is_contiguous():
        raise ValueError("GPTQ weight and permutation must be contiguous")
    if q_weight.device != q_perm.device:
        raise ValueError("GPTQ weight and permutation must be on the same device")
    if q_weight.data_ptr() % 8 or q_perm.data_ptr() % 4:
        raise ValueError("GPTQ uint64 weight and int32 permutation accesses must be aligned")


@label("gptq_shuffle")
@parametrize("q_weight_bit", [
    # bit=8: verified stable combos
    ((128, 512), 8),
    ((256, 1024), 8),
    ((1024, 4096), 8),
    # bit=4: verified stable combos
    ((32, 256), 4),
    ((512, 4096), 4),
    ((1024, 4096), 4),
])
def test_accuracy_gptq_shuffle(q_weight_bit):
    # ===== Accuracy Test =====
    rows, width, unpacked_rows = _gptq_layout(q_weight_bit)
    bit = q_weight_bit[1]
    dtype = torch.int32
    # vLLM 0.13 launches its permutation kernel on the current stream but
    # calls cudaMemcpyAsync without a stream argument. A non-default PyTorch
    # stream can race that copy; fail explicitly rather than run unsafely.
    if torch.cuda.current_stream(device).cuda_stream != torch.cuda.default_stream(device).cuda_stream:
        raise RuntimeError("vLLM 0.13 gptq_shuffle requires the default CUDA stream")

    # Create inputs
    # q_weight: packed int32 tensor on CUDA
    q_weight_init = torch.randint(
        low=0, high=2**31 - 1,
        size=(rows, width),
        device=device, dtype=dtype,
    )
    # vLLM reads exactly 32 / bit indices per packed row and uses each
    # index to address an unpacked row. randperm guarantees all are in range.
    q_perm_tensor = torch.randperm(unpacked_rows, device=device, dtype=torch.int32)

    # Clone inputs for baseline and triton runs
    q_weight_ref = q_weight_init.clone()
    q_weight_act = q_weight_init.clone()
    _check_gptq_buffers(q_weight_ref, q_perm_tensor, rows, width, unpacked_rows)
    _check_gptq_buffers(q_weight_act, q_perm_tensor, rows, width, unpacked_rows)

    # Call baseline: kernelgenbench.baseline.gptq_shuffle(...)
    kernelgenbench.baseline.gptq_shuffle(q_weight_ref, q_perm_tensor, bit)

    # Call triton:   kernelgenbench.triton.gptq_shuffle(...)
    kernelgenbench.triton.gptq_shuffle(q_weight_act, q_perm_tensor, bit)

    # Compare: int32 output, use exact comparison
    assert torch.equal(q_weight_act, q_weight_ref), f"Mismatch: max diff={(q_weight_act - q_weight_ref).abs().max()}"

    # ===== Performance Test =====
    # Prepare fresh data for benchmarking
    q_weight_baseline = torch.randint(
        low=0, high=2**31 - 1,
        size=(rows, width),
        device=device, dtype=dtype,
    )
    q_weight_triton = q_weight_baseline.clone()
    q_perm_bench = torch.randperm(unpacked_rows, device=device, dtype=torch.int32)
    _check_gptq_buffers(q_weight_baseline, q_perm_bench, rows, width, unpacked_rows)
    _check_gptq_buffers(q_weight_triton, q_perm_bench, rows, width, unpacked_rows)

    # Benchmark baseline
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.gptq_shuffle(q_weight_baseline, q_perm_bench, bit),
        warmup=25, rep=100
    )

    # Benchmark triton
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.gptq_shuffle(q_weight_triton, q_perm_bench, bit),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
