"""Regression tests for cuBLAS pointer-mode switching."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kernelgenbench.dataset.baseline.cublas import _backend


def test_nvidia_pointer_mode_switch_is_not_ignored(monkeypatch):
    fn = Mock(return_value=0)
    backend = SimpleNamespace(vendor=SimpleNamespace(value='nvidia'),
                              blas_set_pointer_mode_fn='cublasSetPointerMode_v2')
    monkeypatch.setattr(_backend, 'get_backend', lambda: backend)
    monkeypatch.setattr(_backend, 'get_blas_lib', lambda: SimpleNamespace(cublasSetPointerMode_v2=fn))
    monkeypatch.setattr(_backend, '_spm_fn', None)
    _backend.set_pointer_mode(123, 0)
    _backend.set_pointer_mode(123, 1)
    assert fn.call_args_list[0].args == (123, 0)
    assert fn.call_args_list[1].args == (123, 1)


def test_pointer_mode_switch_propagates_failure(monkeypatch):
    fn = Mock(return_value=7)
    backend = SimpleNamespace(vendor=SimpleNamespace(value='nvidia'),
                              blas_set_pointer_mode_fn='cublasSetPointerMode_v2')
    monkeypatch.setattr(_backend, 'get_backend', lambda: backend)
    monkeypatch.setattr(_backend, 'get_blas_lib', lambda: SimpleNamespace(cublasSetPointerMode_v2=fn))
    monkeypatch.setattr(_backend, '_spm_fn', None)
    with pytest.raises(RuntimeError, match='status 7'):
        _backend.set_pointer_mode(123, 0)
