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

import builtins
import importlib
import json
from collections import OrderedDict
from types import SimpleNamespace

import kernelgenbench
import pytest

verifier_module = importlib.import_module("sandbox.verifier.verifier")
register_module = importlib.import_module("sandbox.register")


def test_guard_preserves_reference_and_accepts_only_same_candidate(monkeypatch):
    monkeypatch.setattr(register_module, "REGISTERED_OPS", OrderedDict())
    monkeypatch.setattr(kernelgenbench, "baseline", SimpleNamespace(), raising=False)
    monkeypatch.setattr(kernelgenbench, "triton", SimpleNamespace(), raising=False)
    original_module_register = register_module.register
    original_public_register = kernelgenbench.register

    def official():
        return "official"

    def candidate():
        return "candidate"

    def different_candidate():
        return "different"

    original_module_register("example", "example", namespace="baseline")(official)
    with verifier_module.candidate_registration_guard("example"):
        kernelgenbench.register("example", "example", namespace="baseline")(
            different_candidate
        )
        assert register_module.REGISTERED_OPS["baseline"]["example"][1] is official
        kernelgenbench.register("example", "example", namespace="triton")(candidate)
        register_module.register("example", "example", namespace="triton")(candidate)
        with pytest.raises(ValueError, match="Conflicting candidate registration"):
            register_module.register("example", "example", namespace="triton")(
                different_candidate
            )

    assert register_module.register is original_module_register
    assert kernelgenbench.register is original_public_register
    assert kernelgenbench.baseline.example is official
    assert kernelgenbench.triton.example is candidate


def test_guard_restores_registration_after_exception():
    original_module_register = register_module.register
    original_public_register = kernelgenbench.register
    with pytest.raises(RuntimeError, match="abort"):
        with verifier_module.candidate_registration_guard("example"):
            raise RuntimeError("abort")
    assert register_module.register is original_module_register
    assert kernelgenbench.register is original_public_register


def test_guard_preserves_explicit_backward_value(monkeypatch):
    monkeypatch.setattr(register_module, "REGISTERED_OPS", OrderedDict())
    monkeypatch.setattr(kernelgenbench, "triton", SimpleNamespace(), raising=False)

    def candidate():
        pass

    with verifier_module.candidate_registration_guard("example"):
        register_module.register("example", "example", None, "triton")(candidate)
    assert register_module.REGISTERED_OPS["triton"]["example"][2] is None


def test_completed_cases_survive_interruption_without_old_progress(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    partial = tmp_path / "report.json.partial.jsonl"
    partial.write_text('{"stale": true}\n', encoding="utf-8")

    def first_case():
        pass

    def interrupted_case():
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        verifier_module, "get_funcs_by_label", lambda _: [(first_case, None), (interrupted_case, None)]
    )
    monkeypatch.setattr(verifier_module, "get_params", lambda *_: None)
    runner = SimpleNamespace(_running_config=SimpleNamespace(anti_hack=False))

    with pytest.raises(KeyboardInterrupt):
        verifier_module.Verifier.run_tests(runner, "example", json_path=str(report))

    cases = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()]
    assert cases == [
        {"function": "first_case", "params": {}, "success": True, "traceback": None}
    ]
    assert not report.exists()


def test_checkpoint_io_failure_does_not_change_official_result(tmp_path, monkeypatch):
    def passing_case():
        pass

    monkeypatch.setattr(verifier_module, "get_funcs_by_label", lambda _: [(passing_case, None)])
    monkeypatch.setattr(verifier_module, "get_params", lambda *_: None)
    real_open = builtins.open

    def fail_partial_file(path, *args, **kwargs):
        if str(path).endswith(".partial.jsonl"):
            raise OSError("checkpoint unavailable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_partial_file)
    runner = SimpleNamespace(_running_config=SimpleNamespace(anti_hack=False))
    result = verifier_module.Verifier.run_tests(runner, "example", json_path=str(tmp_path / "report.json"))
    assert result.success is True
    assert result.info["total"] == 1


def test_temporary_partial_file_is_removed_after_success():
    paths = []
    expected_result = object()

    def run_tests(name, json_path, **kwargs):
        paths.append(json_path)
        with open(json_path + ".partial.jsonl", "w", encoding="utf-8") as partial:
            partial.write('{"success": true}\n')
        return expected_result

    runner = SimpleNamespace(run_tests=run_tests)
    config = SimpleNamespace(save_log=False, seed=42, strict_check=True)
    assert verifier_module.Verifier.run_test(runner, "example", config) is expected_result
    assert not verifier_module.os.path.exists(paths[0])
    assert not verifier_module.os.path.exists(paths[0] + ".partial.jsonl")
