"""CPU smoke checks for repaired ATen fixtures in an isolated process.

The test suite module constructs tensors during import.  A child process avoids
changing sandbox.config.DEVICE or label registries in the main pytest process.
These checks do not replace the full H20 parameter sweep.
"""
import os
import subprocess
import sys
import textwrap


def test_repaired_aten_cpu_scenarios():
    code = textwrap.dedent('''
        import torch
        import sandbox.config as config
        config.DEVICE = 'cpu'
        from kernelgenbench.accuracy import test_ops_with_benchmark as suite

        class Bench:
            @staticmethod
            def do_bench(fn, **kwargs):
                fn()
                return (1.0, 1.0, 1.0)

        suite.device = 'cpu'
        suite.get_triton_testing = lambda: Bench
        scenarios = [
            (suite.test_accuracy_bmm, (3, 4, 5, torch.float32)),
            *[(suite.test_accuracy_cumsum, ((3, 4), dtype))
              for dtype in (torch.float32, torch.int16, torch.int32)],
            *[(suite.test_accuracy_div_scalar_scalar, (dtype,))
              for dtype in (torch.float32, torch.int64)],
            (suite.test_accuracy_trunc_div, ((2, 3), torch.float32)),
            (suite.test_accuracy_trunc_div_, ((2, 3), torch.float32)),
            *[(suite.test_embedding, (16, 2, 3, 5, padding, freq, torch.float32))
              for padding in (None, -1, 1) for freq in (False, True)],
            *[(suite.test_accuracy_mm, (3, 4, 5, torch.float32, layout))
              for layout in (False, True)],
            (suite.test_accuracy_pow_tensor_scalar, (0.5, (3, 4), torch.float32)),
            (suite.test_accuracy_index, ((4, 5), ((2, 3), (3,)), torch.float32)),
            (suite.test_index_put__acc_true, ((4, 5), ((3,),), (3, 5), False, torch.float32)),
            (suite.test_index_put__accumulate_repeated_indices, (torch.float32,)),
        ]
        for fn, args in scenarios:
            fn(*args)

        # A biased candidate must fail bmm's independent float64 oracle.
        state = {'candidate': False}
        real_bmm = suite.torch.bmm
        class CandidateContext:
            def __enter__(self):
                state['candidate'] = True
            def __exit__(self, *args):
                state['candidate'] = False
        suite.kernelgenbench.use_ops = lambda *args, **kwargs: CandidateContext()
        suite.torch.bmm = lambda a, b: real_bmm(a, b) + (0.003 if state['candidate'] else 0.0)
        try:
            suite.test_accuracy_bmm(2, 3, 1024, torch.float32)
        except AssertionError:
            pass
        else:
            raise AssertionError('biased bmm candidate was not rejected')

        def mutating_bmm(a, b):
            result = real_bmm(a, b)
            if state['candidate']:
                a.zero_()
            return result
        suite.torch.bmm = mutating_bmm
        try:
            suite.test_accuracy_bmm(2, 3, 16, torch.float32)
        except AssertionError:
            pass
        else:
            raise AssertionError('input-mutating bmm candidate was not rejected')

        # The empty-index contract must exercise both reference and candidate;
        # an internal upstream assertion must remain a visible failure.
        calls = []
        def expected_index_error(inp, indices):
            calls.append(state['candidate'])
            raise IndexError('at least one index must be provided')
        suite.torch.ops.aten.index = expected_index_error
        suite.test_index_error_empty_indices(torch.float32)
        assert calls == [False, True]
        def broken_reference(inp, indices):
            raise RuntimeError('INTERNAL ASSERT FAILED')
        suite.torch.ops.aten.index = broken_reference
        try:
            suite.test_index_error_empty_indices(torch.float32)
        except RuntimeError as exc:
            assert 'INTERNAL ASSERT FAILED' in str(exc)
        else:
            raise AssertionError('upstream internal assertion was hidden')
        print(f'ATEN_CPU_SCENARIOS_PASSED={len(scenarios)}')
    ''')
    env = dict(os.environ, DISPATCH_TORCH_LIB='0')
    process = subprocess.run([sys.executable, '-c', code], env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert process.returncode == 0, process.stdout + '\n' + process.stderr
    assert 'ATEN_CPU_SCENARIOS_PASSED=20' in process.stdout
