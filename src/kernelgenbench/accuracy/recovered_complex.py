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

"""Recover the historical complex view-resolution accuracy cases.

Source: repository commit 5d47094,
``src/flagbench/accuracy/test_v2_1_ops_with_benchmark.py``, lines 10982-11044.
The cases retain ``SPECIAL_SHAPES`` and complex64, but construct lazy views
on the actual test device and check values, metadata and input preservation.
This module supplies accuracy tests only; it does not report speedups.
"""
import torch

import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.register import REGISTERED_OPS
from sandbox.utils.accuracy_utils import SPECIAL_SHAPES, assert_equal
from sandbox.verifier.test_parametrize import label, parametrize


@label("resolve_conj")
@parametrize("shape", SPECIAL_SHAPES)
@parametrize("dtype", [torch.complex64])
def test_accuracy_resolve_conj(shape, dtype):
    # Form views after allocation on the target device: moving an already
    # conjugated CPU tensor to CUDA may silently materialize its conjugation.
    ref_base = torch.randn(shape, dtype=dtype, device=device)
    act_base = ref_base.clone()
    ref_view = ref_base.conj()
    act_view = act_base.conj()
    assert ref_view.is_conj() and act_view.is_conj(), "inputs must be lazy conjugate views"
    assert not ref_view.is_neg() and not act_view.is_neg()
    assert ref_view.device == act_view.device == ref_base.device

    # Compute an independent materialized oracle without resolving either view.
    oracle = torch.conj_physical(ref_base)
    ref_out = torch.resolve_conj(ref_view)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        act_out = torch.resolve_conj(act_view)

    assert ref_out.dtype == act_out.dtype == dtype
    assert ref_out.shape == act_out.shape == ref_view.shape
    assert ref_out.device == act_out.device == ref_base.device
    assert not ref_out.is_conj() and not act_out.is_conj()
    assert not ref_out.is_neg() and not act_out.is_neg()
    assert ref_view.is_conj() and act_view.is_conj(), "resolution changed input metadata"
    assert_equal(ref_out, oracle)
    assert_equal(act_out, oracle)
    assert_equal(act_base, ref_base)


@label("resolve_neg")
@parametrize("shape", SPECIAL_SHAPES)
@parametrize("dtype", [torch.complex64])
def test_accuracy_resolve_neg(shape, dtype):
    # The imaginary component of a lazy conjugate is a lazy negative view.
    # Construct both independent views on the target device before resolving.
    ref_base = torch.randn(shape, dtype=dtype, device=device)
    act_base = ref_base.clone()
    ref_conj = ref_base.conj()
    act_conj = act_base.conj()
    assert ref_conj.is_conj() and act_conj.is_conj()
    ref_view = ref_conj.imag
    act_view = act_conj.imag
    assert ref_view.is_neg() and act_view.is_neg(), "inputs must be lazy negative views"
    assert not ref_view.is_conj() and not act_view.is_conj()
    assert ref_view.device == act_view.device == ref_base.device

    oracle = torch.neg(ref_base.imag)
    ref_out = torch.resolve_neg(ref_view)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        act_out = torch.resolve_neg(act_view)

    assert ref_out.dtype == act_out.dtype == ref_view.dtype == torch.float32
    assert ref_out.shape == act_out.shape == ref_view.shape
    assert ref_out.device == act_out.device == ref_base.device
    assert not ref_out.is_neg() and not act_out.is_neg()
    assert not ref_out.is_conj() and not act_out.is_conj()
    assert ref_view.is_neg() and act_view.is_neg(), "resolution changed input metadata"
    assert_equal(ref_out, oracle)
    assert_equal(act_out, oracle)
    assert_equal(act_base, ref_base)
