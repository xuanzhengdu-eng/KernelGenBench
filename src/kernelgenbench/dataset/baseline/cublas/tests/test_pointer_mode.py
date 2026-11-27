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


def test_reused_handle_tracks_current_torch_stream(monkeypatch):
    import torch

    fn = Mock(return_value=0)
    backend = SimpleNamespace(vendor=SimpleNamespace(value='nvidia'))
    monkeypatch.setattr(_backend, 'get_backend', lambda: backend)
    monkeypatch.setattr(_backend, 'get_blas_lib', lambda: SimpleNamespace(cublasSetStream_v2=fn))
    monkeypatch.setattr(_backend, '_stream_fn', None)
    monkeypatch.setattr(_backend, '_handle', 123)
    streams = iter([101, 202])
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: SimpleNamespace(cuda_stream=next(streams)))
    assert _backend.get_or_create_handle() == 123
    assert _backend.get_or_create_handle() == 123
    assert [(call.args[0], call.args[1].value) for call in fn.call_args_list] == [(123, 101), (123, 202)]


def test_stream_binding_propagates_failure(monkeypatch):
    import torch

    fn = Mock(return_value=7)
    backend = SimpleNamespace(vendor=SimpleNamespace(value='nvidia'))
    monkeypatch.setattr(_backend, 'get_blas_lib', lambda: SimpleNamespace(cublasSetStream_v2=fn))
    monkeypatch.setattr(_backend, '_stream_fn', None)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: SimpleNamespace(cuda_stream=101))
    with pytest.raises(RuntimeError, match='status 7'):
        _backend._bind_current_cuda_stream(123, backend)
