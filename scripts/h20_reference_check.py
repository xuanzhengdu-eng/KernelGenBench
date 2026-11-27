#!/usr/bin/env python3
"""Run KernelGenBench accuracy tests with reference implementations on both sides.

This is an environment/coverage diagnostic, not an independent correctness or
performance benchmark. One subprocess per operator isolates CUDA errors.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['DISPATCH_TORCH_LIB'] = '0'


def worker(op, directory):
    import torch
    import kernelgenbench
    from sandbox.verifier import Verifier, VerifyConfig

    category, name = op.split('::', 1)
    cfg = VerifyConfig(run_name='ref', run_dir=str(directory), strict_check=True,
                       anti_hack=False, manage_device_visibility=False)
    verifier = Verifier(cfg)
    if category == 'aten':
        importlib.import_module('kernelgenbench.accuracy.test_ops_with_benchmark')
    else:
        if category == 'vllm13':
            # Do not silently treat an importable wrapper with missing custom ops as a pass.
            importlib.import_module('vllm._custom_ops')
        baseline = importlib.import_module('kernelgenbench.dataset.baseline.' + category)
        implementation = getattr(baseline, name)
        from types import SimpleNamespace
        kernelgenbench.baseline = SimpleNamespace(**{name: implementation})
        kernelgenbench.triton = SimpleNamespace(**{name: implementation})
        importlib.import_module('kernelgenbench.accuracy.' + category)
    from sandbox.verifier.test_parametrize import get_funcs_by_label, get_params
    import math
    funcs = get_funcs_by_label(name)
    matching = [(f, mark) for f, mark in funcs if category in f.__module__]
    if matching:
        funcs = matching
    expected = sum(math.prod(len(slot['values']) for slot in get_params(f.__name__, mark))
                   for f, mark in funcs)
    print('device:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    print('registered functions:', len(funcs), 'expected cases:', expected, flush=True)
    result = verifier.run_tests(op, json_path=str(directory / 'cases.json'),
                                max_failures='all', seed=42, strict_check=True)
    data = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
    data['expected_cases'] = expected
    data['registered_functions'] = len(funcs)
    actual = (data.get('info') or {}).get('total', 0)
    data['coverage_complete'] = expected > 0 and actual == expected
    if not data['coverage_complete']:
        data['status'] = 'incomplete_coverage'
        if data['success'] is True:
            data['success'] = None
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--operator', action='append', help='Operator(s), e.g. aten::add; default: all 210')
    parser.add_argument('--category', action='append', choices=['aten', 'cublas', 'vllm13'],
                        help='Run only the specified categories; may be repeated')
    parser.add_argument('--timeout', type=int, default=1800, help='Seconds per operator')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if args.worker:
        try:
            result = worker(args.worker, out)
            (out / 'result.json').write_text(json.dumps(result, indent=2, default=str))
        except Exception:
            (out / 'result.json').write_text(json.dumps({'op_name': args.worker, 'success': False,
                'status': 'setup_error', 'traceback': traceback.format_exc()}, indent=2))
            raise
        return
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    from kernelgenbench.dataset.kernel_list import KERNELGENBENCH_OPERATOR_NAMES
    if args.operator and args.category:
        parser.error('--operator and --category cannot be combined')
    ops = args.operator or ([op for op in KERNELGENBENCH_OPERATOR_NAMES
                             if op.split('::', 1)[0] in args.category]
                            if args.category else KERNELGENBENCH_OPERATOR_NAMES)
    invalid = [op for op in ops if op not in KERNELGENBENCH_OPERATOR_NAMES]
    if invalid:
        parser.error(f'Unknown operators: {invalid}')
    if len(set(ops)) != len(ops):
        parser.error('Duplicate operators')
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'operators': ops, 'device_visibility': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'timeout_seconds': args.timeout, 'note': 'ref vs ref; not independent validation or performance'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    results = []
    for index, op in enumerate(ops, 1):
        # Preserve disk capacity for other users on the shared host.
        if shutil.disk_usage(out).free < 5 * 1024**3:
            print('Stopping: less than 5 GiB free on output filesystem', flush=True)
            break
        op_dir = out / op.replace('::', '__')
        op_dir.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', op, '--output', str(op_dir)]
        print(f'[{index}/{len(ops)}] {op}', flush=True)
        with (op_dir / 'worker.log').open('w') as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                result_path = op_dir / 'result.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    result['worker_exit_code'] = process.returncode
                else:
                    result = {'op_name': op, 'success': False, 'status': 'worker_exit', 'worker_exit_code': process.returncode}
            except subprocess.TimeoutExpired:
                result = {'op_name': op, 'success': None, 'status': 'timeout'}
        results.append(result)
        (out / 'summary.json').write_text(json.dumps({'requested': len(ops), 'completed': len(results),
            'passed': sum(r.get('success') is True for r in results),
            'failed': sum(r.get('success') is False for r in results),
            'no_result': sum(r.get('success') is None for r in results), 'results': results}, indent=2, default=str))
        print(f'  result: {result.get("success")}, {result.get("status", "finished")}', flush=True)


if __name__ == '__main__':
    main()
