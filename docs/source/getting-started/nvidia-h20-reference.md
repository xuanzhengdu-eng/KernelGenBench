# NVIDIA H20 reference self-check

This diagnostic runs the dataset's existing test functions with the reference
implementation on **both** sides of the comparison. It checks the test and
runtime setup; it does **not** independently validate reference correctness or
measure kernel speedup. No model service or API key is needed.

H20-3e on the tested host reports CUDA compute capability 9.0. The existing
NVIDIA backend applies; no separate vendor backend is required. For a host
whose CUDA installation does not contain `/usr/local/cuda/lib64/libcublas.so.12`,
a CUDA 12 container can supply the expected library without changing the host.

The following example uses a locally available `vllm/vllm-openai:v0.12.0` image
with PyTorch 2.9 and CUDA 12.9 for ATen and cuBLAS only. It is **not** the
project's requested vLLM 0.13 baseline: use a separate compatible vLLM 0.13
environment for `--category vllm13`. Install pytest in an isolated directory
and mount it read-only; do not install into another user's running container.
Replace paths and GPU number as appropriate, and ensure the GPU is available.

```bash
# Run inside a compatible Python/CUDA environment, with this repo's src on PYTHONPATH.
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/pytest-deps:/path/to/KernelGenBench/src
python scripts/h20_reference_check.py --output /path/to/new-results \
  --category aten --category cublas --timeout 1800

# In the separate vLLM 0.13 environment, after the GPU becomes available:
python scripts/h20_reference_check.py --output /path/to/new-vllm-results \
  --category vllm13 --timeout 1800
```

Omit `--category` to select all 210 operators in an environment satisfying all
requirements, or use `--operator aten::erfc` for one operator. Output directories
must not already exist. The runner isolates each operator in a subprocess,
records full case results and tracebacks, and marks timeouts, setup errors, and
incomplete coverage explicitly. It stops starting new operators if output
storage falls below 5 GiB. The upstream tests also contain performance loops;
those loops still execute and the full run may take a long time. Ref-versus-ref
performance numbers are diagnostic noise and must not be presented as speedups.

For a Docker container exposing only physical GPU 6, CUDA typically sees that
device as `cuda:0`; pass `--gpus device=6` to Docker and set
`CUDA_VISIBLE_DEVICES=0` inside. Avoid overlapping other workloads. Keep the
test container network disabled and do not expose a vLLM service: old vLLM
versions are used solely for reproducing the benchmark baseline.
