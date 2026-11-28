"""CPU-only vLLM fixture checks isolated from the main pytest process."""
import os
import subprocess
import sys
import textwrap


def test_vllm_fixture_contracts_cpu():
    code = textwrap.dedent('''
        import importlib
        from types import SimpleNamespace
        import torch
        import sandbox.config as config
        config.DEVICE = 'cpu'
        cache_test = importlib.import_module('kernelgenbench.accuracy.vllm13.test_cp_gather_indexer_k_quant_cache')
        gptq_test = importlib.import_module('kernelgenbench.accuracy.vllm13.test_gptq_shuffle')
        assert gptq_test._gptq_layout(((128, 512), 8)) == (128, 128, 512)
        assert gptq_test._gptq_layout(((1024, 4096), 4)) == (1024, 512, 8192)
        for cfg in (((0, 256), 4), ((32, 255), 4), ((32, 256), 3)):
            try:
                gptq_test._gptq_layout(cfg)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid GPTQ layout accepted: {cfg}')

        def gather(cache, out_k, out_s, block_table, cu_seq):
            _, block_size, _ = cache.shape
            head_dim, scale_bytes = out_k.shape[1], out_s.shape[1]
            for batch in range(block_table.shape[0]):
                start, end = int(cu_seq[batch]), int(cu_seq[batch + 1])
                for offset in range(end - start):
                    block = int(block_table[batch, offset // block_size])
                    slot = offset % block_size
                    raw = cache[block].view(-1)
                    out_k[start + offset].view(torch.uint8).copy_(raw[slot * head_dim:(slot + 1) * head_dim])
                    scale_start = block_size * head_dim + slot * scale_bytes
                    out_s[start + offset].copy_(raw[scale_start:scale_start + scale_bytes])

        cache_test.device = 'cpu'
        cache_test.triton.testing.do_bench = lambda fn, **kw: (fn(), 1.0)[1]
        cache_test.kernelgenbench = SimpleNamespace(
            baseline=SimpleNamespace(cp_gather_indexer_k_quant_cache=gather),
            triton=SimpleNamespace(cp_gather_indexer_k_quant_cache=gather))
        config_case = (4, 16, 4, 64, 2)
        cache_test.test_accuracy_cp_gather_indexer_k_quant_cache(config_case)
        def wrong(cache, out_k, out_s, table, cu_seq):
            gather(cache, out_k, out_s, table, cu_seq)
            out_k.view(torch.uint8)[0, 0] ^= 1
        cache_test.kernelgenbench.triton.cp_gather_indexer_k_quant_cache = wrong
        try:
            cache_test.test_accuracy_cp_gather_indexer_k_quant_cache(config_case)
        except AssertionError as exc:
            assert 'candidate K bytes' in str(exc)
        else:
            raise AssertionError('independent cache oracle accepted an incorrect candidate')

        # A candidate that ignores the physical block table must also fail.
        def wrong_block_table(cache, out_k, out_s, table, cu_seq):
            identity = torch.arange(table.numel(), dtype=table.dtype).reshape_as(table).remainder(cache.shape[0])
            gather(cache, out_k, out_s, identity, cu_seq)
        cache_test.kernelgenbench.triton.cp_gather_indexer_k_quant_cache = wrong_block_table
        try:
            cache_test.test_accuracy_cp_gather_indexer_k_quant_cache(config_case)
        except AssertionError as exc:
            assert 'candidate K bytes' in str(exc)
        else:
            raise AssertionError('candidate ignoring block mapping was not rejected')
        print('VLLM_FIXTURE_SCENARIOS_PASSED=8')
    ''')
    env = dict(os.environ, DISPATCH_TORCH_LIB='0')
    process = subprocess.run([sys.executable, '-c', code], env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert process.returncode == 0, process.stdout + '\n' + process.stderr
    assert 'VLLM_FIXTURE_SCENARIOS_PASSED=8' in process.stdout
