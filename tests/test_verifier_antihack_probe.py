# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Regression coverage for anti-hack probe selection and failure reporting."""

import importlib
from types import SimpleNamespace

verifier_module = importlib.import_module("sandbox.verifier.verifier")
anti_hack = importlib.import_module("sandbox.anti_hack")


def _run(monkeypatch, case, combos):
    monkeypatch.setattr(verifier_module, "get_funcs_by_label", lambda _: [(case, None)])
    monkeypatch.setattr(verifier_module, "get_params", lambda *_: object())
    monkeypatch.setattr(verifier_module, "expand_params", lambda _: combos)
    runner = SimpleNamespace(_running_config=SimpleNamespace(anti_hack=True))
    return verifier_module.Verifier.run_tests(runner, "example")


def test_anti_hack_uses_executed_case_and_its_own_normal_output(monkeypatch):
    checked = []

    def test_case(shape):
        if not shape:
            return None
        return {"shape": list(shape)}

    def dual(func, kwargs, output):
        checked.append(("dual", kwargs, output))
        return False, ""

    def profile(func, kwargs):
        checked.append(("profile", kwargs))
        return False, ""

    monkeypatch.setattr(anti_hack, "dual_execution_check_with_ref", dual)
    monkeypatch.setattr(anti_hack, "gpu_profiling_check", profile)
    result = _run(monkeypatch, test_case, [{"shape": ()}, {"shape": (1,)}, {"shape": (4,)}])

    assert result.success is True
    assert result.info == {"total": 3, "failed": 0, "success": 3}
    assert checked == [("dual", {"shape": (1,)}, {"shape": [1]}), ("profile", {"shape": (1,)})]


def test_profile_detection_still_fails_and_reports_the_reason(monkeypatch):
    checked = []

    def test_case(shape):
        return {"ok": True} if shape else None

    monkeypatch.setattr(anti_hack, "dual_execution_check_with_ref", lambda *_: (False, ""))
    def profile(func, kwargs):
        checked.append(kwargs)
        return True, "GPU profiling hack detected: no CUDA kernels launched at all."
    monkeypatch.setattr(anti_hack, "gpu_profiling_check", profile)

    result = _run(monkeypatch, test_case, [{"shape": ()}, {"shape": (1,)}])
    assert checked == [{"shape": (1,)}]
    assert result.success is False
    assert result.info["failed"] == 2
    assert "no CUDA kernels launched" in result.traceback


def test_all_none_results_still_run_the_existing_checks(monkeypatch):
    checked = []

    def test_case(value):
        return None

    def dual(func, kwargs):
        checked.append(("dual", kwargs))
        return False, ""

    def profile(func, kwargs):
        checked.append(("profile", kwargs))
        return True, "no CUDA kernels launched"

    monkeypatch.setattr(anti_hack, "dual_execution_check", dual)
    monkeypatch.setattr(anti_hack, "gpu_profiling_check", profile)
    result = _run(monkeypatch, test_case, [{"value": 1}, {"value": 2}])

    assert checked == [("dual", {"value": 1}), ("profile", {"value": 1})]
    assert result.success is False
    assert "no CUDA kernels launched" in result.traceback


def test_cuda_copies_do_not_disguise_native_kernels_as_triton(monkeypatch):
    monkeypatch.setattr(
        anti_hack,
        "profile_kernel_names",
        lambda *_: (["void at::native::fill_kernel", "Memcpy DtoD (Device -> Device)", "Memcpy DtoH (Device -> Pinned)"], []),
    )
    detected, reason = anti_hack.gpu_profiling_check(lambda: None, {})
    assert detected is True
    assert "all launched kernels are native" in reason


def test_cuda_copy_alone_does_not_count_as_a_kernel(monkeypatch):
    monkeypatch.setattr(anti_hack, "profile_kernel_names", lambda *_: (["Memcpy DtoD (Device -> Device)"], []))
    detected, reason = anti_hack.gpu_profiling_check(lambda: None, {})
    assert detected is True
    assert "no CUDA kernels launched" in reason


def test_real_triton_kernel_is_not_discarded_with_cuda_copies(monkeypatch):
    monkeypatch.setattr(anti_hack, "profile_kernel_names", lambda *_: (["Memcpy DtoD (Device -> Device)", "narrow_kernel"], []))
    detected, reason = anti_hack.gpu_profiling_check(lambda: None, {})
    assert detected is False
    assert reason == ""
