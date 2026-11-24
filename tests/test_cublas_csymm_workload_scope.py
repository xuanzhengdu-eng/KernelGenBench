"""Guard the full A100 cublasCsymm_v2 workload."""
from collections import Counter

from kernelgenbench.accuracy.cublas.test_cublasCsymm_v2 import (
    test_accuracy_cublasCsymm_v2 as csymm_accuracy_case,
)
from sandbox.verifier.test_parametrize import get_params
from sandbox.verifier.utils import expand_params


def test_csymm_workload_retains_all_parameter_combinations():
    cases = expand_params(get_params(csymm_accuracy_case, "default"))
    assert len(cases) == 60
    assert Counter(case["n"] for case in cases) == {
        1: 12, 32: 12, 64: 12, 128: 12, 160: 12
    }
    assert {case["alpha"] for case in cases} == {1.0, 0.0, -1.0}
    assert {case["beta"] for case in cases} == {0.0, 0.5}
    assert {case["side"] for case in cases} == {0, 1}
