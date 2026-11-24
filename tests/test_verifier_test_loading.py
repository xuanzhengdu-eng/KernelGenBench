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

import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

parametrize_module = importlib.import_module("sandbox.verifier.test_parametrize")
verifier_module = importlib.import_module("sandbox.verifier.verifier")
expand_params = importlib.import_module("sandbox.verifier.utils").expand_params


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch):
    monkeypatch.setattr(parametrize_module, "_label_registry", {})
    monkeypatch.setattr(parametrize_module, "_parameter_registry", {})
    monkeypatch.setattr(parametrize_module, "_function_parameter_registry", {})


def test_distinct_functions_with_same_name_keep_their_own_parameters():
    parametrize = parametrize_module.parametrize

    @parametrize("value", [1, 2])
    def same_name(value):
        pass

    first = same_name

    @parametrize("value", [3, 4, 5])
    def same_name(value):
        pass

    assert len(expand_params(parametrize_module.get_params(first))) == 2
    assert len(expand_params(parametrize_module.get_params(same_name))) == 3


def test_stacked_parameters_preserve_repeated_values():
    parametrize = parametrize_module.parametrize

    @parametrize("x", [1, 1])
    @parametrize("y", [2, 3, 4])
    def example(x, y):
        pass

    combinations = expand_params(parametrize_module.get_params(example))
    assert len(combinations) == 6
    assert combinations.count({"x": 1, "y": 2}) == 2
    assert parametrize_module.get_params("example") == parametrize_module.get_params(example)


def test_package_eager_import_and_directory_loader_register_file_once(tmp_path, monkeypatch):
    package = tmp_path / "kgb_test_loader_package"
    package.mkdir()
    (package / "__init__.py").write_text("from . import test_example\n", encoding="utf-8")
    module = package / "test_example.py"
    module.write_text(
        "from sandbox.verifier.test_parametrize import label, parametrize\n"
        "@label('loader_example')\n"
        "@parametrize('value', [1, 2, 3])\n"
        "def test_example(value): pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    loader = object.__new__(verifier_module.Verifier)
    loader._import_module_or_path(str(package))
    loader._import_module_or_path(str(module))
    funcs = parametrize_module.get_funcs_by_label("loader_example")
    assert len(funcs) == 1
    assert funcs[0][0].__module__ == "kgb_test_loader_package.test_example"
    assert len(expand_params(parametrize_module.get_params(funcs[0][0]))) == 3
    sys.modules.pop("kgb_test_loader_package.test_example", None)
    sys.modules.pop("kgb_test_loader_package", None)


def test_alias_imports_do_not_repeat_cases_or_merge_same_named_definitions(tmp_path):
    module = tmp_path / "test_repeated.py"
    module.write_text(
        "from sandbox.verifier.test_parametrize import label, parametrize\n"
        "@label('same_label')\n"
        "@parametrize('value', [1, 2])\n"
        "def test_same(value): pass\n"
        "@label('same_label')\n"
        "@parametrize('value', [3, 4, 5])\n"
        "def test_same(value): pass\n",
        encoding="utf-8",
    )
    for alias in ("kgb_alias_one", "kgb_alias_two"):
        spec = importlib.util.spec_from_file_location(alias, module)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)

    assert len(parametrize_module.get_funcs_by_label("same_label")) == 4
    runner = SimpleNamespace(_running_config=SimpleNamespace(anti_hack=False))
    result = verifier_module.Verifier.run_tests(runner, "same_label")
    assert result.success is True
    assert result.info["total"] == 5


def test_different_external_files_with_same_basename_are_both_loaded(tmp_path):
    loader = object.__new__(verifier_module.Verifier)
    for i, values in enumerate(("[1, 2]", "[3, 4, 5]")):
        directory = tmp_path / str(i)
        directory.mkdir()
        module = directory / "test_same.py"
        module.write_text(
            "from sandbox.verifier.test_parametrize import label, parametrize\n"
            "@label('external_same')\n"
            f"@parametrize('value', {values})\n"
            "def test_same(value): pass\n",
            encoding="utf-8",
        )
        loader._import_module_or_path(str(module))
    runner = SimpleNamespace(_running_config=SimpleNamespace(anti_hack=False))
    result = verifier_module.Verifier.run_tests(runner, "external_same")
    assert result.success is True
    assert result.info["total"] == 5


def test_package_name_resolving_to_other_file_is_rejected(tmp_path, monkeypatch):
    paths = []
    for i in range(2):
        root = tmp_path / str(i)
        package = root / "kgb_conflicting_package"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        module = package / "test_example.py"
        module.write_text(f"VALUE = {i}\n", encoding="utf-8")
        paths.append(module)
        monkeypatch.syspath_prepend(str(root))
    loader = object.__new__(verifier_module.Verifier)
    loader._import_module_or_path(str(paths[1]))
    with pytest.raises(ImportError, match="resolved to a different file"):
        loader._import_module_or_path(str(paths[0]))
    sys.modules.pop("kgb_conflicting_package.test_example", None)
    sys.modules.pop("kgb_conflicting_package", None)


def test_loaded_module_scan_does_not_call_dynamic_getattr(tmp_path, monkeypatch):
    trap = ModuleType("kgb_scan_trap")

    def fail_getattr(name):
        raise AssertionError(f"Unexpected module attribute access: {name}")

    trap.__getattr__ = fail_getattr
    monkeypatch.setitem(sys.modules, "kgb_scan_trap", trap)
    module = tmp_path / "independent_test.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    object.__new__(verifier_module.Verifier)._import_module_or_path(str(module))
