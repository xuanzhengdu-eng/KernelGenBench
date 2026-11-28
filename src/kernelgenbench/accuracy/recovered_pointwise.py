# Copyright 2026 FlagOS Contributors
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

"""Restore three pointwise test labels from FlagBench history.

Derived from commit 5d47094, test_v2_1_ops_with_benchmark.py:9264-9319 and
9433-9501. Historical shapes, dtypes, fill values and tensor/scalar overloads
are preserved. The exponential test additionally checks two distributions
statistically; it deliberately does not require two RNG implementations to
produce identical elementwise samples.

The existing ATen accuracy module imports this module to register its labels
for both the normal verifier and the reference self-check.
"""
import math

import kernelgenbench
import torch

from sandbox.config import DEVICE as device
from sandbox.register import REGISTERED_OPS
from sandbox.utils.accuracy_utils import (
    CustomBenchmarkResult,
    DISTRIBUTION_SHAPES,
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    kernelgenbench_assert_equal,
    to_reference,
)
from sandbox.verifier.test_parametrize import label, parametrize


_EXP_SEED = 20261128


def _inplace_contract(tensor, result, shape, stride, dtype, device_before, data_ptr):
    """Check that the same input object and storage were modified in place."""
    assert result is tensor, 'in-place operation must return its input object'
    assert tensor.shape == shape, 'in-place operation changed input shape'
    assert tensor.stride() == stride, 'in-place operation changed input stride'
    assert tensor.dtype == dtype, 'in-place operation changed input dtype'
    assert tensor.device == device_before, 'in-place operation changed input device'
    assert tensor.data_ptr() == data_ptr, 'in-place operation changed input storage'


def _benchmark(reference_fn, candidate_fn, ref_device):
    """Preserve the historical GPU benchmark; CPU-only correctness has no timing."""
    if torch.device(ref_device).type != 'cuda' or torch.device(device).type != 'cuda':
        return None
    from triton.testing import do_bench

    quantiles = [0.5, 0.2, 0.8]
    ref_ms, _, _ = do_bench(reference_fn, rep=100, quantiles=quantiles)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        act_ms, _, _ = do_bench(candidate_fn, rep=100, quantiles=quantiles)
    return CustomBenchmarkResult(ref_time=ref_ms, res_time=act_ms,
                                 speedup=ref_ms / act_ms if act_ms > 0 else float('inf'))


def _sample_moments_and_cdf(tensor):
    values = tensor.detach().to(dtype=torch.float32)
    assert bool(torch.isfinite(values).all().item()), 'exponential output must be finite'
    assert bool((values > 0).all().item()), 'historical exponential test requires positive output'
    return (values.mean().item(), values.square().mean().item(),
            [(values <= cutoff).to(dtype=torch.float32).mean().item()
             for cutoff in (0.5, 1.0, 2.0)])


def _check_exponential_distribution(reference, actual, dtype):
    """Check Exp(1) moments/CDF and two-sample agreement with conservative bounds.

    The mean has variance 1/n; E[X^2]=2 and Var(X^2)=20/n. Each CDF
    estimate is Bernoulli with variance F(t)*(1-F(t))/n. Seven standard
    errors and dtype-epsilon allowances avoid requiring matching samples.
    These checks are statistical, not proof of a particular RNG sequence.
    """
    assert reference.numel() == actual.numel() and actual.numel() > 0
    count = actual.numel()
    eps = torch.finfo(dtype).eps
    ref_mean, ref_second, ref_cdf = _sample_moments_and_cdf(reference)
    act_mean, act_second, act_cdf = _sample_moments_and_cdf(actual)
    mean_bound = 7 / math.sqrt(count) + 2 * eps
    second_bound = 7 * math.sqrt(20 / count) + 8 * eps
    assert abs(ref_mean - 1.0) <= mean_bound, f'reference exponential mean {ref_mean}'
    assert abs(act_mean - 1.0) <= mean_bound, f'candidate exponential mean {act_mean}'
    assert abs(ref_second - 2.0) <= second_bound, f'reference second moment {ref_second}'
    assert abs(act_second - 2.0) <= second_bound, f'candidate second moment {act_second}'
    assert abs(ref_mean - act_mean) <= 7 * math.sqrt(2 / count) + 4 * eps, 'exponential means differ'
    assert abs(ref_second - act_second) <= 7 * math.sqrt(40 / count) + 16 * eps, 'exponential second moments differ'
    for cutoff, ref_value, act_value in zip((0.5, 1.0, 2.0), ref_cdf, act_cdf):
        expected = 1.0 - math.exp(-cutoff)
        variance = expected * (1.0 - expected) / count
        single_bound = 7 * math.sqrt(variance) + 2 * eps
        pair_bound = 7 * math.sqrt(2 * variance) + 4 * eps
        assert abs(ref_value - expected) <= single_bound, f'reference CDF at {cutoff}: {ref_value}'
        assert abs(act_value - expected) <= single_bound, f'candidate CDF at {cutoff}: {act_value}'
        assert abs(ref_value - act_value) <= pair_bound, f'exponential CDFs differ at {cutoff}'


def _seed_default_generator(tensor):
    """Seed only the relevant CPU/CUDA default generator inside fork_rng."""
    if tensor.device.type == 'cpu':
        torch.random.default_generator.manual_seed(_EXP_SEED)
    elif tensor.device.type == 'cuda':
        index = tensor.device.index
        if index is None:
            index = torch.cuda.current_device()
        torch.cuda.default_generators[index].manual_seed(_EXP_SEED)
    else:
        raise RuntimeError(f'exponential RNG restoration is not implemented for {tensor.device.type}')


@label('eq')
@parametrize('shape', POINTWISE_SHAPES)
@parametrize('dtype', FLOAT_DTYPES)
def test_recovered_eq_tensor(shape, dtype):
    """Historical eq.Tensor shapes and float dtypes; inputs must remain unchanged."""
    left = torch.randint(0, 10, shape, dtype=dtype, device=device)
    right = torch.randint(0, 10, shape, dtype=dtype, device=device)
    ref_left, ref_right = to_reference(left.clone()), to_reference(right.clone())
    ref_out = torch.eq(ref_left, ref_right)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        act_out = torch.eq(left, right)
    assert act_out.dtype == torch.bool and act_out.shape == ref_out.shape
    kernelgenbench_assert_equal(act_out, ref_out)
    kernelgenbench_assert_equal(left, ref_left)
    kernelgenbench_assert_equal(right, ref_right)
    return _benchmark(lambda: torch.eq(ref_left.clone(), ref_right.clone()),
                      lambda: torch.eq(left.clone(), right.clone()), ref_left.device)


@label('eq')
@parametrize('shape', POINTWISE_SHAPES)
@parametrize('dtype', FLOAT_DTYPES)
def test_recovered_eq_scalar(shape, dtype):
    """Historical eq.Scalar overload uses scalar zero."""
    left = torch.randint(0, 10, shape, dtype=dtype, device=device)
    ref_left = to_reference(left.clone())
    ref_out = torch.eq(ref_left, 0)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        act_out = torch.eq(left, 0)
    assert act_out.dtype == torch.bool and act_out.shape == ref_out.shape
    kernelgenbench_assert_equal(act_out, ref_out)
    kernelgenbench_assert_equal(left, ref_left)
    return _benchmark(lambda: torch.eq(ref_left.clone(), 0),
                      lambda: torch.eq(left.clone(), 0), ref_left.device)


@label('exponential_')
@parametrize('shape', DISTRIBUTION_SHAPES)
@parametrize('dtype', FLOAT_DTYPES)
def test_recovered_exponential_(shape, dtype):
    """Historical default-rate exponential_ with isolated reproducible RNG state."""
    act = torch.full(shape, -1.0, dtype=dtype, device=device)
    ref = to_reference(act.clone())
    cuda_devices = sorted({t.device.index if t.device.index is not None else torch.cuda.current_device()
                           for t in (ref, act) if t.device.type == 'cuda'})
    ref_meta = (ref.shape, ref.stride(), ref.dtype, ref.device, ref.data_ptr())
    act_meta = (act.shape, act.stride(), act.dtype, act.device, act.data_ptr())
    # fork_rng saves/restores CPU and the selected CUDA generators. The two
    # invocations receive the same seed but only distributions are compared.
    with torch.random.fork_rng(devices=cuda_devices):
        _seed_default_generator(ref)
        ref_ret = ref.exponential_()
        _seed_default_generator(act)
        with kernelgenbench.use_ops(REGISTERED_OPS):
            act_ret = act.exponential_()
        _inplace_contract(ref, ref_ret, *ref_meta)
        _inplace_contract(act, act_ret, *act_meta)
        _check_exponential_distribution(ref, act, dtype)
        # In reference-only mode both sides use the same backend and seed;
        # require exact reproducibility. Generated kernels remain subject to
        # distribution checks rather than an identical RNG algorithm.
        if not kernelgenbench.DISPATCH_TORCH_LIB:
            kernelgenbench_assert_equal(act, ref)
        return _benchmark(lambda: ref.clone().exponential_(),
                          lambda: act.clone().exponential_(), ref.device)


@label('fill_')
@parametrize('value', [0, 1, 9])
@parametrize('shape', POINTWISE_SHAPES)
@parametrize('dtype', FLOAT_DTYPES)
def test_recovered_fill_(value, shape, dtype):
    """Exercise both historical fill_.Scalar and fill_.Tensor overloads."""
    for tensor_value in (False, True):
        act = torch.ones(shape, dtype=dtype, device=device)
        ref = to_reference(act.clone())
        scalar = torch.tensor(value, dtype=dtype, device=device) if tensor_value else value
        ref_scalar = to_reference(scalar.clone()) if tensor_value else value
        ref_meta = (ref.shape, ref.stride(), ref.dtype, ref.device, ref.data_ptr())
        act_meta = (act.shape, act.stride(), act.dtype, act.device, act.data_ptr())
        ref_ret = ref.fill_(ref_scalar)
        with kernelgenbench.use_ops(REGISTERED_OPS):
            act_ret = act.fill_(scalar)
        _inplace_contract(ref, ref_ret, *ref_meta)
        _inplace_contract(act, act_ret, *act_meta)
        kernelgenbench_assert_equal(act, ref)
        if tensor_value:
            kernelgenbench_assert_equal(scalar, ref_scalar)
    # The historical benchmark covers the scalar overload.
    ref_bench = to_reference(torch.ones(shape, dtype=dtype, device=device))
    act_bench = torch.ones(shape, dtype=dtype, device=device)
    return _benchmark(lambda: ref_bench.clone().fill_(value),
                      lambda: act_bench.clone().fill_(value), ref_bench.device)
