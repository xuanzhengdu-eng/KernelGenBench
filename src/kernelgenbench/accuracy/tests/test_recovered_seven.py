"""CPU-only isolation tests for recovered historical ATen registrations."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_recovered_seven_registration_and_semantics():
    code = textwrap.dedent('''
        import math
        import torch
        import sandbox.config as config
        config.DEVICE = 'cpu'
        import kernelgenbench
        import kernelgenbench.accuracy.test_ops_with_benchmark
        from kernelgenbench.accuracy import recovered_pointwise as pointwise
        from kernelgenbench.accuracy import recovered_views as views
        from kernelgenbench.accuracy import recovered_complex as complex_ops
        from sandbox.verifier.test_parametrize import get_funcs_by_label, get_params
        expected = {'eq': (2, 36), 'exponential_': (1, 3), 'fill_': (1, 54),
                    'expand': (3, 21), 'expand_as': (2, 18),
                    'resolve_conj': (1, 5), 'resolve_neg': (1, 5)}
        for label, (num_functions, num_cases) in expected.items():
            funcs = get_funcs_by_label(label)
            cases = sum(math.prod(len(slot['values']) for slot in get_params(f.__name__, mark))
                        for f, mark in funcs)
            assert (len(funcs), cases) == (num_functions, num_cases), (label, len(funcs), cases)
        assert len(get_funcs_by_label('add')) > 0
        pointwise.test_recovered_eq_tensor((2, 3), torch.float32)
        pointwise.test_recovered_eq_scalar((2, 3), torch.float32)
        pointwise.test_recovered_fill_(9, (2, 3), torch.float32)
        pointwise.test_recovered_exponential_((20, 320, 15), torch.float32)
        views.test_recovered_expand((1, 1, 32), (4, 8, 32), torch.float32)
        views.test_recovered_expand_minus_one(torch.float32)
        views.test_recovered_expand_as((1, 16), (8, 16), torch.float32)
        views.test_recovered_expand_noncontiguous_offset(torch.float32)
        views.test_recovered_expand_as_noncontiguous_offset(torch.float32)
        complex_ops.test_accuracy_resolve_conj((2, 3), torch.complex64)
        complex_ops.test_accuracy_resolve_neg((2, 3), torch.complex64)

        # A wrong candidate must be observed under dispatch and rejected.
        active = {'candidate': False}
        class CandidateContext:
            def __enter__(self):
                active['candidate'] = True
            def __exit__(self, *args):
                active['candidate'] = False
        kernelgenbench.use_ops = lambda *args, **kwargs: CandidateContext()
        real_eq = torch.eq
        def wrong_eq(left, right):
            result = real_eq(left, right)
            return torch.logical_not(result) if active['candidate'] else result
        torch.eq = wrong_eq
        try:
            pointwise.test_recovered_eq_tensor((2, 3), torch.float32)
        except AssertionError:
            pass
        else:
            raise AssertionError('wrong eq candidate was not rejected')
        print('RECOVERED_LABELS=7 CASES=142 NEGATIVE_CONTROL=PASS')
    ''')
    env = dict(os.environ, DISPATCH_TORCH_LIB='0')
    result = subprocess.run([sys.executable, '-c', code], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert 'RECOVERED_LABELS=7 CASES=142 NEGATIVE_CONTROL=PASS' in result.stdout


def test_verifier_explicit_file_entrypoint_loads_recovered_labels(tmp_path):
    code = textwrap.dedent('''
        import sys
        import sandbox.config as config
        config.DEVICE = 'cpu'
        from sandbox.verifier import Verifier, VerifyConfig
        from sandbox.verifier.test_parametrize import get_funcs_by_label
        verifier = Verifier(VerifyConfig(run_name='entrypoint', run_dir=sys.argv[2], anti_hack=False))
        verifier.set_modules([sys.argv[1]], mode='accuracy')
        verifier.import_tests('accuracy')
        expected = {'eq': 2, 'exponential_': 1, 'fill_': 1, 'expand': 3,
                    'expand_as': 2, 'resolve_conj': 1, 'resolve_neg': 1}
        for label, count in expected.items():
            assert len(get_funcs_by_label(label)) == count, (label, get_funcs_by_label(label))
        print('EXPLICIT_FILE_ENTRYPOINT_OK')
    ''')
    repo = Path(__file__).resolve().parents[4]
    module = repo / 'src/kernelgenbench/accuracy/test_ops_with_benchmark.py'
    env = dict(os.environ, DISPATCH_TORCH_LIB='0')
    result = subprocess.run([sys.executable, '-c', code, str(module), str(tmp_path)],
                            env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert 'EXPLICIT_FILE_ENTRYPOINT_OK' in result.stdout
