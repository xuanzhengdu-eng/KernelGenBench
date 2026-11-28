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

"""Recovered ATen view tests from FlagBench v2.1, commit 5d47094.

Historical source: ``src/flagbench/accuracy/test_v2_1_ops_with_benchmark.py``
lines 9325-9428.  The original shape/dtype and ``-1`` cases are retained;
additional non-contiguous, nonzero-storage-offset cases check view semantics.
These are accuracy tests, not performance benchmarks: the historical timing
included clone costs and would not isolate the metadata-only view operation.
"""
import torch
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.register import REGISTERED_OPS
from sandbox.utils.accuracy_utils import FLOAT_DTYPES, kernelgenbench_assert_equal, to_reference
from sandbox.verifier.test_parametrize import label, parametrize


# Exact shape pairs from the historical expand and expand_as tests.
_VIEW_SHAPES = [
    ((1,), (8,)),
    ((1, 16), (8, 16)),
    ((1, 1, 32), (4, 8, 32)),
    ((16, 1), (16, 32)),
    ((1, 16, 1), (8, 16, 32)),
]


def _snapshot(tensor):
    """Record values and metadata before entering the candidate dispatch."""
    return (
        tensor.detach().clone(), tuple(tensor.shape), tuple(tensor.stride()),
        tensor.storage_offset(), tensor.untyped_storage().data_ptr(),
        tensor.dtype, tensor.device,
    )


def _assert_unchanged(tensor, before):
    values, shape, stride, offset, storage_ptr, dtype, original_device = before
    assert tuple(tensor.shape) == shape, 'input shape changed'
    assert tuple(tensor.stride()) == stride, 'input stride changed'
    assert tensor.storage_offset() == offset, 'input storage offset changed'
    assert tensor.untyped_storage().data_ptr() == storage_ptr, 'input storage changed'
    assert tensor.dtype == dtype and tensor.device == original_device, 'input dtype/device changed'
    torch.testing.assert_close(tensor, values, rtol=0, atol=0)


def _assert_view_pair(ref_base, ref_out, ref_before, act_base, act_out, act_before, dtype):
    """Compare result data and enforce actual PyTorch expand view behavior."""
    _assert_unchanged(ref_base, ref_before)
    _assert_unchanged(act_base, act_before)
    assert ref_out.dtype == dtype and act_out.dtype == dtype, 'output dtype mismatch'
    assert ref_out.device == ref_base.device and act_out.device == act_base.device, 'output device mismatch'
    assert tuple(act_out.shape) == tuple(ref_out.shape), 'output shape mismatch'
    assert tuple(act_out.stride()) == tuple(ref_out.stride()), 'output stride mismatch'
    assert act_out.storage_offset() == ref_out.storage_offset(), 'output storage offset mismatch'
    assert act_out.is_contiguous() == ref_out.is_contiguous(), 'output contiguity mismatch'
    for base, view in ((ref_base, ref_out), (act_base, act_out)):
        assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr(), 'expand must alias input storage'
        assert view.storage_offset() == base.storage_offset(), 'expand must preserve the input storage offset'
    kernelgenbench_assert_equal(act_out, ref_out)

    # A write to one expanded element must be visible through the input.  Use
    # a single index (not a bulk write to overlapping zero-stride elements).
    output_index = (0,) * ref_out.ndim
    input_index = (0,) * ref_base.ndim
    ref_expected = ref_before[0].clone()
    act_expected = act_before[0].clone()
    marker = -13.25 if ref_expected[input_index].item() == 13.25 else 13.25
    ref_expected[input_index] = marker
    act_expected[input_index] = marker
    ref_out[output_index] = marker
    act_out[output_index] = marker
    torch.testing.assert_close(ref_base, ref_expected, rtol=0, atol=0)
    torch.testing.assert_close(act_base, act_expected, rtol=0, atol=0)


def _inputs(shape, dtype):
    original = torch.randn(shape, dtype=dtype, device=device)
    # to_reference may move to CPU, but never rely on it to create a copy.
    return to_reference(original.clone()), original.clone()


@label('expand')
@parametrize('shape,expand_shape', _VIEW_SHAPES)
@parametrize('dtype', FLOAT_DTYPES)
def test_recovered_expand(shape, expand_shape, dtype):
    ref_base, act_base = _inputs(shape, dtype)
    ref_before, act_before = _snapshot(ref_base), _snapshot(act_base)
    ref_out = ref_base.expand(expand_shape)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        act_out = act_base.expand(expand_shape)
    _assert_view_pair(ref_base, ref_out, ref_before, act_base, act_out, act_before, dtype)


@label('expand')
@parametrize('dtype', FLOAT_DTYPES)
def test_recovered_expand_minus_one(dtype):
    ref_base, act_base = _inputs((4, 1, 8), dtype)
    ref_before, act_before = _snapshot(ref_base), _snapshot(act_base)
    ref_out = ref_base.expand(-1, 16, -1)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        act_out = act_base.expand(-1, 16, -1)
    _assert_view_pair(ref_base, ref_out, ref_before, act_base, act_out, act_before, dtype)


@label('expand_as')
@parametrize('shape,other_shape', _VIEW_SHAPES)
@parametrize('dtype', FLOAT_DTYPES)
def test_recovered_expand_as(shape, other_shape, dtype):
    ref_base, act_base = _inputs(shape, dtype)
    ref_other, act_other = _inputs(other_shape, dtype)
    ref_before, act_before = _snapshot(ref_base), _snapshot(act_base)
    ref_other_before, act_other_before = _snapshot(ref_other), _snapshot(act_other)
    ref_out = ref_base.expand_as(ref_other)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        act_out = act_base.expand_as(act_other)
    _assert_unchanged(ref_other, ref_other_before)
    _assert_unchanged(act_other, act_other_before)
    _assert_view_pair(ref_base, ref_out, ref_before, act_base, act_out, act_before, dtype)
    _assert_unchanged(ref_other, ref_other_before)
    _assert_unchanged(act_other, act_other_before)


def _offset_inputs(dtype):
    # Both inputs are views with shape (1, 5), stride (1, 4), and offset 1.
    parent = torch.randn((5, 4), dtype=dtype, device=device)
    reference_parent = to_reference(parent.clone())
    return reference_parent.transpose(0, 1)[1:2, :], parent.transpose(0, 1)[1:2, :]


@label('expand')
@parametrize('dtype', FLOAT_DTYPES)
def test_recovered_expand_noncontiguous_offset(dtype):
    ref_base, act_base = _offset_inputs(dtype)
    ref_before, act_before = _snapshot(ref_base), _snapshot(act_base)
    ref_out = ref_base.expand(3, 5)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        act_out = act_base.expand(3, 5)
    _assert_view_pair(ref_base, ref_out, ref_before, act_base, act_out, act_before, dtype)


@label('expand_as')
@parametrize('dtype', FLOAT_DTYPES)
def test_recovered_expand_as_noncontiguous_offset(dtype):
    ref_base, act_base = _offset_inputs(dtype)
    ref_other, act_other = _inputs((3, 5), dtype)
    ref_before, act_before = _snapshot(ref_base), _snapshot(act_base)
    ref_other_before, act_other_before = _snapshot(ref_other), _snapshot(act_other)
    ref_out = ref_base.expand_as(ref_other)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        act_out = act_base.expand_as(act_other)
    _assert_unchanged(ref_other, ref_other_before)
    _assert_unchanged(act_other, act_other_before)
    _assert_view_pair(ref_base, ref_out, ref_before, act_base, act_out, act_before, dtype)
    _assert_unchanged(ref_other, ref_other_before)
    _assert_unchanged(act_other, act_other_before)
