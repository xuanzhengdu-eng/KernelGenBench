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
from kernelgenbench.accuracy.vllm13._ggml import make_q8_0_data


def _make_q8_0_data(rows, cols):
    return make_q8_0_data((rows,), cols, device)


@label("ggml_dequantize")
@parametrize("config", [
    # (rows, cols, quant_type, dtype)
    # --- Small: accuracy-only ---
    (32, 128, 8, torch.float16),
    (64, 256, 8, torch.float16),
    # --- Medium: accuracy + perf ---
    (128, 512, 8, torch.float16),
    (256, 1024, 8, torch.float32),
    (512, 2048, 8, torch.float16),
    # --- Large / real model weight shapes: accuracy + perf ---
    (1024, 4096, 8, torch.float16),    # Llama-7B hidden
    (4096, 4096, 8, torch.float16),    # Llama-7B square weight
    (4096, 11008, 8, torch.float32),   # Llama-7B FFN weight
])
def test_accuracy_ggml_dequantize(config):
    # ===== Accuracy Test =====
    rows, cols, quant_type, dtype = config

    W = _make_q8_0_data(rows, cols)

    ref_out = kernelgenbench.baseline.ggml_dequantize(W, quant_type, rows, cols, dtype)
    act_out = kernelgenbench.triton.ggml_dequantize(W, quant_type, rows, cols, dtype)

    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    if rows * cols < 65536:
        return None

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.ggml_dequantize(W, quant_type, rows, cols, dtype),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.ggml_dequantize(W, quant_type, rows, cols, dtype),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
