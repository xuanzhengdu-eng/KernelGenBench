#!/usr/bin/env python3
"""Validate a private SM90 AllSpark build against independent PyTorch math.

Also checks two non-default CUDA streams. This is a small correctness test,
not an operator performance benchmark or proof for every possible shape.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', default=os.environ.get('KERNELGENBENCH_H20_ALLSPARK_LIBRARY'),
                        help='Built kg_allspark_h20.so; alternatively set KERNELGENBENCH_H20_ALLSPARK_LIBRARY')
    args = parser.parse_args()
    if not args.library:
        parser.error('--library or KERNELGENBENCH_H20_ALLSPARK_LIBRARY is required')
    if os.environ.get('CUDA_LAUNCH_BLOCKING') == '1':
        raise RuntimeError('CUDA_LAUNCH_BLOCKING=1 would serialize the multi-stream check')
    if torch.__version__.split('+')[0] != '2.9.0' or torch.version.cuda != '12.9':
        raise RuntimeError('Validation requires the tested PyTorch 2.9.0/CUDA 12.9 ABI')
    device = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(device)
    if (p.major, p.minor) != (9, 0):
        raise RuntimeError(f'Expected SM90 device, got {(p.major, p.minor)}')
    library = Path(args.library).expanduser().resolve(strict=True)
    torch.ops.load_library(str(library))
    runtime = ctypes.CDLL('libcudart.so.12')
    runtime.cudaGetLastError.restype = ctypes.c_int

    def check_launch(stage):
        status = runtime.cudaGetLastError()
        if status:
            raise RuntimeError(f'{stage} CUDA launch failed: {status}')

    def prepare_and_run(m, k, n, dtype, has_zp):
        a = torch.randn((m, k), device='cuda', dtype=dtype)
        q = torch.randint(0, 256, (k, n), device='cuda', dtype=torch.uint8)
        scale = (torch.rand((1, n), device='cuda', dtype=dtype) + 0.01) / 128
        zero = torch.randint(-5, 6, (1, n), device='cuda').to(dtype) if has_zp else None
        padded_n = ((n + 31) // 32) * 32
        rw = torch.empty((padded_n, k), device='cuda', dtype=torch.uint8)
        rs = torch.empty((1, padded_n), device='cuda', dtype=dtype)
        rz = torch.empty_like(rs) if has_zp else None
        torch.ops.kg_allspark_h20.rearrange_kn_weight_as_n32k16_order(
            q, scale, zero, has_zp, rw, rs, rz, k, n, padded_n)
        check_launch('repack')
        out = torch.ops.kg_allspark_h20.allspark_w8a16_gemm(
            a, rw, rs, rz, n, -1, p.multi_processor_count, p.major * 10 + p.minor,
            16, has_zp, True)
        check_launch('gemm')
        return (a, q, scale, zero, rw, rs, rz, out, m, k, n, dtype, has_zp)

    def verify(work, mode):
        a, q, scale, zero, _, _, _, out, m, k, n, dtype, has_zp = work
        weights = q.to(dtype) - 128
        if has_zp:
            weights = weights - zero
        oracle = torch.matmul(a, weights * scale)
        mean = oracle.float().abs().mean().item()
        if mean <= 1e-4 or not bool(torch.isfinite(out).all().item()):
            raise AssertionError('Non-finite output or unexpectedly near-zero reference')
        error = ((out.float() - oracle.float()).abs().mean() / mean).item()
        record = {'mode': mode, 'dtype': str(dtype), 'has_zp': has_zp,
                  'm': m, 'k': k, 'n': n, 'normalized_mean_abs_error': error}
        print(json.dumps(record), flush=True)
        if error >= 0.04:
            raise AssertionError(f'Independent AllSpark reference mismatch: {record}')

    torch.manual_seed(23)
    count = 0
    for dtype in (torch.float16, torch.bfloat16):
        for has_zp in (False, True):
            for m, k, n in ((1, 256, 128), (32, 256, 128), (256, 512, 256)):
                item = prepare_and_run(m, k, n, dtype, has_zp)
                torch.cuda.synchronize()
                verify(item, 'default')
                count += 1
    # Each call retains inputs and outputs until both streams finish; the
    # extension allocates its workspace separately for each invocation.
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    work = []
    for i in range(8):
        dtype = torch.float16 if i % 4 < 2 else torch.bfloat16
        m = 1 if i % 2 == 0 else 32
        with torch.cuda.stream(streams[i % 2]):
            work.append(prepare_and_run(m, 256, 128, dtype, False))
    torch.cuda.synchronize()
    for item in work:
        verify(item, 'nondefault_streams')
    print(f'ALLSPARK_VALIDATED default={count} nondefault_streams={len(work)}', flush=True)


if __name__ == '__main__':
    main()
