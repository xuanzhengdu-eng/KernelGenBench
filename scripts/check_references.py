#!/usr/bin/env python3
"""Exercise the registered benchmark cases with their native reference implementations.

This command checks reference readiness. It does not evaluate submitted kernels or
produce leaderboard scores. The LLM and Agent verification paths remain separate.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Set before importing the benchmark or its dispatch registry.
os.environ["DISPATCH_TORCH_LIB"] = "0"


class ReachedTiming(Exception):
    """The case finished its assertions and reached the Triton timing block."""


@contextmanager
def correctness_only(enabled):
    if not enabled:
        yield
        return
    import triton.testing

    def stop_at_timing(*args, **kwargs):
        raise ReachedTiming

    original = triton.testing.do_bench
    triton.testing.do_bench = stop_at_timing
    try:
        yield
    finally:
        triton.testing.do_bench = original


def get_cases(operator):
    import kernelgenbench
    from sandbox.verifier.test_parametrize import get_funcs_by_label, get_params
    from sandbox.verifier.utils import expand_params

    namespace, name = operator.split("::", 1)
    if namespace == "aten":
        module = importlib.import_module("kernelgenbench.accuracy.test_ops_with_benchmark")
    else:
        module = importlib.import_module(f"kernelgenbench.accuracy.{namespace}.test_{name}")
        baselines = importlib.import_module(f"kernelgenbench.dataset.baseline.{namespace}")
        reference = getattr(baselines, name)
        kernelgenbench.baseline = types.SimpleNamespace(**{name: reference})
        kernelgenbench.triton = types.SimpleNamespace(**{name: reference})
    for function, mark in get_funcs_by_label(name):
        if function.__module__ == module.__name__:
            parameters = get_params(function.__name__, mark)
            for case in expand_params(parameters) if parameters else [{}]:
                yield function, case


def check_operator(operator, args):
    import numpy as np
    import pytest
    import torch

    report = {"operator": operator, "counts": {}, "errors": [], "fatal_cuda": False}
    counts = Counter()
    try:
        cases = list(get_cases(operator))
    except Exception as exc:
        report.update(status="error", errors=[{"error": f"{type(exc).__name__}: {exc}"}])
        return report
    if args.first:
        cases = cases[:1]
    report["expected_cases"] = len(cases)
    if not cases:
        report["status"] = "missing_test"
        return report

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    with correctness_only(args.correctness_only):
        for function, kwargs in cases:
            try:
                result = function(**kwargs)
                torch.cuda.synchronize()
                if result is not None and hasattr(result, "ref_time"):
                    for timing in (result.ref_time, result.res_time):
                        if not math.isfinite(timing) or timing <= 0:
                            raise ValueError(f"Invalid measured time: {timing}")
                    counts["timed"] += 1
                counts["passed"] += 1
            except ReachedTiming:
                # CUDA failures are asynchronous and must be observed before passing.
                try:
                    torch.cuda.synchronize()
                except RuntimeError as exc:
                    counts["failed"] += 1
                    report["errors"].append({"function": function.__name__, "error": str(exc)})
                    report["fatal_cuda"] = True
                    break
                counts.update(passed=1, timing_omitted=1)
            except pytest.skip.Exception as exc:
                counts["skipped"] += 1
                if len(report["errors"]) < 12:
                    report["errors"].append({"function": function.__name__, "skip": str(exc)})
            except Exception as exc:
                counts["failed"] += 1
                if len(report["errors"]) < 12:
                    report["errors"].append({
                        "function": function.__name__,
                        "params": {key: str(value) for key, value in kwargs.items()},
                        "error": f"{type(exc).__name__}: {exc}"[:1500],
                    })
                if "CUDA error:" in str(exc):
                    report["fatal_cuda"] = True
                    break
    report["counts"] = dict(counts)
    report["status"] = "passed" if counts["passed"] == len(cases) else "failed"
    return report


def main():
    from kernelgenbench.dataset import KERNELGENBENCH_OPERATOR_NAMES
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", nargs="+", choices=KERNELGENBENCH_OPERATOR_NAMES)
    parser.add_argument("--shard", nargs=2, type=int, metavar=("INDEX", "COUNT"), default=(0, 1))
    parser.add_argument("--output", required=True, type=Path, help="Write one JSON result per operator here")
    parser.add_argument("--correctness-only", action="store_true", help="Stop at the Triton timing block for triage. Manual timing blocks remain active.")
    parser.add_argument("--first", action="store_true", help="Run the first case of each operator for a smoke test")
    parser.add_argument("--memory-limit-mib", type=int, help="Optional PyTorch allocator cap for shared nodes")
    args = parser.parse_args()
    index, count = args.shard
    if not 0 <= index < count:
        parser.error("shard INDEX must be in [0, COUNT)")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("select a single CUDA GPU through CUDA_VISIBLE_DEVICES")
    if args.memory_limit_mib is not None:
        if args.memory_limit_mib <= 0:
            parser.error("memory limit must be positive")
        fraction = args.memory_limit_mib * 1024**2 / torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(fraction, 1.0), 0)
    args.output.mkdir(parents=True, exist_ok=True)
    operators = args.operators or KERNELGENBENCH_OPERATOR_NAMES
    failed = False
    for operator in operators[index::count]:
        report = check_operator(operator, args)
        report["device"] = torch.cuda.get_device_name(0)
        report["first_case_only"] = args.first
        report["correctness_only"] = args.correctness_only
        target = args.output / (operator.replace("::", "__") + ".json")
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tmp.replace(target)
        print(operator, report["status"], report["counts"], flush=True)
        failed |= report["status"] != "passed"
        if report["fatal_cuda"]:
            return 2
        torch.cuda.empty_cache()
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
