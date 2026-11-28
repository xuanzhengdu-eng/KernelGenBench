#!/usr/bin/env python3
"""Build a private SM90 AllSpark extension from the official vLLM 0.13 source.

This does not replace installed vLLM operators. Pass the resulting library to
KERNELGENBENCH_H20_ALLSPARK_LIBRARY to opt in to the H20 reference adapter.
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import site
import subprocess
import tarfile

SOURCE_SHA256 = '4ad43db45fef37114b550d03a4f423fb3fa3a31d8bc09ee810ef8b9cdcd4b5fe'
SOURCE_FILES = [
    'quantization/gptq_allspark/allspark_qgemm_w8a16.cu',
    'quantization/gptq_allspark/allspark_utils.cuh',
    'quantization/gptq_allspark/allspark_repack.cu',
    'quantization/gptq_marlin/marlin_dtypes.cuh',
    'quantization/gptq_marlin/marlin.cuh',
    'core/scalar_type.hpp',
    'core/registration.h',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-archive', required=True, type=Path,
                        help='Official vllm-0.13.0.tar.gz from PyPI')
    parser.add_argument('--output', required=True, type=Path,
                        help='New directory for sources and compiled library')
    args = parser.parse_args()
    archive = args.source_archive.resolve(strict=True)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f'vLLM 0.13.0 source SHA256 mismatch: {digest}')

    # Fail rather than silently producing an extension for an untested ABI.
    import torch
    if torch.__version__.split('+')[0] != '2.9.0' or torch.version.cuda != '12.9':
        raise RuntimeError(f'Tested build requires torch 2.9.0/CUDA 12.9; found {torch.__version__}/{torch.version.cuda}')
    cuda_home = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')).resolve()
    nvcc = cuda_home / 'bin' / 'nvcc'
    if not nvcc.is_file():
        raise FileNotFoundError(f'CUDA compiler not found under {cuda_home}')
    version_text = subprocess.run([str(nvcc), '--version'], check=True, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    version = re.search(r'release\s+(\d+\.\d+)', version_text)
    if not version or version.group(1) != '12.9':
        raise RuntimeError(f'Tested build requires nvcc 12.9: {version_text.strip()}')
    package_roots = site.getsitepackages()
    include_dirs = sorted({p for root in package_roots for p in glob.glob(f'{root}/nvidia/*/include')})
    lib_dirs = [str(cuda_home / 'lib64')] + [p for root in package_roots
                for p in glob.glob(f'{root}/nvidia/cublas/lib')]
    if not any((Path(p) / 'libcublas.so.12').is_file() for p in lib_dirs):
        raise FileNotFoundError('libcublas.so.12 not found in CUDA or Python NVIDIA package')

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    src, build = root / 'src', root / 'build'
    build.mkdir()
    with tarfile.open(archive, 'r:gz') as tar:
        for rel in SOURCE_FILES:
            member = tar.getmember('vllm-0.13.0/csrc/' + rel)
            if not member.isfile():
                raise RuntimeError(f'Not a regular source entry: {rel}')
            target = src / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())

    # Isolate all helper symbols from the prebuilt SM8x vLLM shared object.
    utils = src / SOURCE_FILES[1]
    utils_text = utils.read_text()
    assert utils_text.count('namespace allspark {') == 1
    utils.write_text(utils_text.replace('namespace allspark {', 'namespace kg_h20_allspark {'))
    cu = src / SOURCE_FILES[0]
    text = cu.read_text()
    assert text.count('namespace allspark {') == 1
    text = text.replace('namespace allspark {', 'namespace kg_h20_allspark {')
    text = text.replace('allspark::', 'kg_h20_allspark::')
    old_global = 'at::Tensor as_g_workspace;'
    old_alloc = '''  if (as_g_workspace.numel() <
      ws_size) {  // ws_options: kChar, so numel() is bytes
    as_g_workspace = torch::empty({long(ws_size)}, ws_options);
  }'''
    assert text.count(old_global) == 1 and text.count(old_alloc) == 1
    text = text.replace(old_global, '// Workspace is allocated per invocation below.')
    text = text.replace(old_alloc, '  auto as_g_workspace = torch::empty({long(ws_size)}, ws_options);')
    assert text.count('torch::Tensor allspark_w8a16_gemm(') == 2
    assert text.count('&allspark_w8a16_gemm') == 1
    text = text.replace('torch::Tensor allspark_w8a16_gemm(', 'torch::Tensor kg_h20_allspark_w8a16_gemm(')
    text = text.replace('&allspark_w8a16_gemm', '&kg_h20_allspark_w8a16_gemm')
    cu.write_text(text)
    repack = src / SOURCE_FILES[2]
    text = repack.read_text()
    assert text.count('namespace allspark {') == 1
    assert text.count('void rearrange_kn_weight_as_n32k16_order(') == 1
    assert text.count('&rearrange_kn_weight_as_n32k16_order') == 1
    text = text.replace('namespace allspark {', 'namespace kg_h20_allspark {')
    text = text.replace('allspark::', 'kg_h20_allspark::')
    text = text.replace('void rearrange_kn_weight_as_n32k16_order(',
                        'void kg_h20_rearrange_kn_weight_as_n32k16_order(')
    text = text.replace('&rearrange_kn_weight_as_n32k16_order',
                        '&kg_h20_rearrange_kn_weight_as_n32k16_order')
    repack.write_text(text)
    schema = src / 'schema.cpp'
    schema.write_text('''#include <torch/library.h>\n#include "core/registration.h"\nTORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {\n  m.def("allspark_w8a16_gemm(Tensor a, Tensor b_qweight, Tensor b_scales, Tensor? b_qzeros, SymInt n, SymInt group_size, SymInt sm_count, SymInt sm_version, SymInt CUBLAS_M_THRESHOLD, bool has_zp, bool n32k16_reorder) -> Tensor");\n  m.def("rearrange_kn_weight_as_n32k16_order(Tensor b_qweight, Tensor b_scales, Tensor? b_zeros, bool has_zp, Tensor! b_qweight_reorder, Tensor! b_scales_reorder, Tensor!? b_zeros_reorder, int K, int N, int N_32align) -> ()");\n}\n''')
    os.environ.setdefault('MAX_JOBS', '1')
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    os.environ['CUDA_HOME'] = str(cuda_home)
    from torch.utils.cpp_extension import load
    library = load(name='kg_allspark_h20', sources=[str(schema), str(cu), str(repack)],
                   extra_include_paths=[str(src), *include_dirs], build_directory=str(build),
                   extra_cflags=['-O2', '-fvisibility=hidden'],
                   extra_cuda_cflags=['-O2', '-Xcompiler=-fvisibility=hidden',
                                      '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
                                      '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__'],
                   extra_ldflags=[*[f'-L{p}' for p in lib_dirs], '-l:libcublas.so.12'],
                   with_cuda=True, is_python_module=False, verbose=True)
    metadata = {'source_sha256': digest, 'architecture': 'sm90', 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'library': str(library), 'workspace': 'per invocation'}
    (root / 'build-metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
