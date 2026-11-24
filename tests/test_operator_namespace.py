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

from pathlib import Path

import pytest

from kernelgenbench.dataset import is_pytorch_op
from sandbox.verifier.verifier import Verifier


@pytest.mark.parametrize(
    "operator",
    [
        "cublas::cublasSdot_v2",
        "vllm13::awq_gemm",
        "unknown_namespace::add",
    ],
)
def test_non_aten_qualified_operators_are_not_pytorch_ops(operator):
    assert is_pytorch_op(operator) is False


@pytest.mark.parametrize("operator", ["add", "aten::add"])
def test_aten_operators_are_still_recognized(operator):
    assert is_pytorch_op(operator) is True


def test_explicit_registration_namespace_is_not_pytorch():
    assert is_pytorch_op("add", namespace="triton") is False


def test_non_aten_name_is_not_misclassified_by_cached_aten_name():
    assert is_pytorch_op("aten::add") is True
    assert is_pytorch_op("cublas::add") is False


@pytest.mark.parametrize(
    ("operator", "function"),
    [
        ("cublas::cublasSdot_v2", "cublasSdot_v2"),
        ("vllm13::awq_gemm", "awq_gemm"),
    ],
)
def test_non_aten_operator_keeps_full_name_for_baseline_lookup(
    operator, function, monkeypatch
):
    # Baselines are UTF-8; avoid depending on the host's default encoding.
    read_text = Path.read_text

    def read_utf8(path, *args, **kwargs):
        if not args and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_utf8)
    assert is_pytorch_op(operator) is False
    baseline = Verifier._auto_load_baseline(None, operator)
    assert baseline is not None
    assert f"def {function}(" in baseline
