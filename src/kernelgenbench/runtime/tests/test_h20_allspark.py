"""CPU tests for the explicitly enabled H20 AllSpark adapter."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from kernelgenbench.runtime import h20_allspark


def test_adapter_is_opt_in(monkeypatch):
    monkeypatch.delenv('KERNELGENBENCH_H20_ALLSPARK_LIBRARY', raising=False)
    assert h20_allspark.h20_allspark_ops(torch.zeros(1)) is None


def test_adapter_rejects_cpu_when_enabled(monkeypatch):
    monkeypatch.setenv('KERNELGENBENCH_H20_ALLSPARK_LIBRARY', '/not/a/library')
    with pytest.raises(RuntimeError, match='SM90'):
        h20_allspark.h20_allspark_ops(torch.zeros(1))


def test_adapter_rejects_unverified_torch_abi(monkeypatch):
    monkeypatch.setenv('KERNELGENBENCH_H20_ALLSPARK_LIBRARY', '/not/a/library')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (9, 0))
    monkeypatch.setattr(torch, '__version__', '2.10.0')
    with pytest.raises(RuntimeError, match='PyTorch 2.9.0/CUDA 12.9'):
        h20_allspark.h20_allspark_ops(SimpleNamespace(is_cuda=True, device='cuda:0'))


def test_adapter_loads_explicit_path_on_sm90(monkeypatch, tmp_path):
    library = tmp_path / 'example.so'
    library.write_bytes(b'fixture')
    sentinel = object()
    monkeypatch.setenv('KERNELGENBENCH_H20_ALLSPARK_LIBRARY', str(library))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (9, 0))
    monkeypatch.setattr(torch, '__version__', '2.9.0')
    monkeypatch.setattr(torch.version, 'cuda', '12.9')
    monkeypatch.setattr(h20_allspark, '_load_library', lambda path: sentinel if path == str(library) else None)
    assert h20_allspark.h20_allspark_ops(SimpleNamespace(is_cuda=True, device='cuda:0')) is sentinel


def test_repack_validates_contract_and_passes_expected_shapes(monkeypatch):
    calls = []
    def repack(q, s, z, has_zp, q_out, s_out, z_out, k, n, padded_n):
        calls.append((q.shape, s.shape, has_zp, q_out.shape, s_out.shape, k, n, padded_n))
        q_out.zero_()
        s_out.zero_()
    monkeypatch.setattr(h20_allspark, 'h20_allspark_ops', lambda tensor: SimpleNamespace(
        rearrange_kn_weight_as_n32k16_order=repack))
    q = torch.zeros((32, 48), dtype=torch.uint8)
    s = torch.ones((1, 48), dtype=torch.float16)
    q_out, s_out, z_out = h20_allspark.allspark_repack_weight(q, s)
    assert q_out.shape == (64, 32) and s_out.shape == (1, 64) and z_out is None
    assert calls == [(q.shape, s.shape, False, q_out.shape, s_out.shape, 32, 48, 64)]
    with pytest.raises(ValueError, match='K and N'):
        h20_allspark.allspark_repack_weight(torch.zeros((0, 48), dtype=torch.uint8), s)
    with pytest.raises(ValueError, match='scale'):
        h20_allspark.allspark_repack_weight(q, torch.ones((1, 47), dtype=torch.float16))
    with pytest.raises(ValueError, match='zero_point'):
        h20_allspark.allspark_repack_weight(q, s, None, True)


def test_builder_rejects_unverified_source_before_writing(tmp_path):
    root = Path(__file__).resolve().parents[4]
    archive = tmp_path / 'untrusted.tar.gz'
    archive.write_bytes(b'not the official vllm source')
    output = tmp_path / 'build'
    result = subprocess.run([sys.executable, str(root / 'scripts/build_h20_allspark.py'),
                             '--source-archive', str(archive), '--output', str(output)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert result.returncode != 0 and 'source SHA256 mismatch' in result.stderr
    assert not output.exists()
