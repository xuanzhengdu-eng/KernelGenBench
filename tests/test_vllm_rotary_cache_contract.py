"""Regression for the vLLM rotary benchmark's cache layout."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("is_neox", [False, True])
def test_rotary_benchmark_passes_full_width_cache(monkeypatch, is_neox):
    pytest.importorskip("vllm")
    from vllm import _custom_ops
    from kernelgenbench.accuracy.vllm13 import test_rotary_embedding as benchmark

    seen = []

    def reference(positions, query, key, head_size, cache, style):
        assert tuple(cache.shape) == (16, head_size)
        seen.append(tuple(cache.shape))
        _custom_ops.rotary_embedding(positions, query, key, head_size, cache, style)

    # Use the actual reference in both slots: this test isolates the benchmark
    # input contract, independently of a particular Triton candidate.
    monkeypatch.setattr(benchmark.kernelgenbench, "baseline", SimpleNamespace(rotary_embedding=reference), raising=False)
    monkeypatch.setattr(benchmark.kernelgenbench, "triton", SimpleNamespace(rotary_embedding=reference), raising=False)
    benchmark.test_accuracy_rotary_embedding((4, 16), 2, 32, torch.float32, False, is_neox)
    assert seen == [(16, 32), (16, 32)]
