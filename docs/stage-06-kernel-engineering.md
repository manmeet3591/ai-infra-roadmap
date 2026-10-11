# Stage 6 — Kernel-Level Engineering

> **"Python overhead kills inference. Fused kernels save milliseconds that compound at scale."**

**Status:** ⏳ Pending

---

## What to Learn

- **Triton** — Python-based GPU kernel language from OpenAI
- **FlashAttention** — memory-efficient attention via tiling
- **CUDA graphs** — eliminating CPU launch overhead for repeated workloads
- **Operator fusion** — combining multiple ops into a single kernel pass

---

## Core Concepts

### Why Kernel Fusion Matters

```
Unfused: each op reads/writes to HBM separately
Input → [RMSNorm] → HBM → [Linear] → HBM → [SiLU] → HBM → Output
Total memory traffic: 8 × tensor_size

Fused kernel: intermediates stay in registers/shared memory
Input → [Fused: RMSNorm + Linear + SiLU] → Output
Total memory traffic: 2 × tensor_size
Speedup: 4×+
```

### FlashAttention — The Key Idea

```
Standard attention O(N²) memory:
[Q][K][V] → [S: N×N] → [P: N×N] → [O]  (N² grows fast!)

FlashAttention O(N) memory:
Process Q, K, V in tiles that fit in SRAM.
Never materialize full N×N matrix in HBM.
Result: 2–4× speedup, O(N) memory.
```

### Triton Basics

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

import torch
x = torch.randn(1_000_000, device='cuda')
y = torch.randn(1_000_000, device='cuda')
output = torch.empty_like(x)
n = x.numel()
BLOCK_SIZE = 1024
grid = (triton.cdiv(n, BLOCK_SIZE),)
add_kernel[grid](x, y, output, n, BLOCK_SIZE)
```

### CUDA Graphs

```python
import torch

# Warm up
for _ in range(3):
    outputs = model(**inputs)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(**inputs)

# Replay with zero CPU overhead
for _ in range(1000):
    g.replay()
```

---

## Practice Project: Fused RMSNorm in Triton

```python
import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    x_sq_mean = tl.sum(x * x, axis=0) / n_cols
    rms = tl.sqrt(x_sq_mean + eps)
    x_norm = x / rms
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    tl.store(output_ptr + row_start + offsets, x_norm * weight, mask=mask)

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    rmsnorm_kernel[(n_rows,)](x, weight, output, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output

def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() / rms * weight.float()).to(x.dtype)

def benchmark(fn, *args, warmup=10, iters=100):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    for _ in range(iters): fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000

N_ROWS, N_COLS = 4096, 4096
x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float16)
w = torch.ones(N_COLS, device='cuda', dtype=torch.float16)

assert torch.allclose(torch_rmsnorm(x, w), triton_rmsnorm(x, w), atol=1e-2)

t_torch = benchmark(torch_rmsnorm, x, w)
t_triton = benchmark(triton_rmsnorm, x, w)
print(f"PyTorch: {t_torch:.3f} ms")
print(f"Triton:  {t_triton:.3f} ms")
print(f"Speedup: {t_torch/t_triton:.2f}×")
```

---

## Results Log

| Kernel | Shape | PyTorch (ms) | Triton (ms) | Speedup |
|--------|-------|-------------|------------|--------|
| RMSNorm | 4096×4096 | — | — | — |
| Softmax | 4096×4096 | — | — | — |
| FlashAttention vs sdpa | seq=8192, d=128 | — | — | — |

---

## Resources

- [Triton Documentation](https://triton-lang.org/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [liger-kernel](https://github.com/linkedin/Liger-Kernel) — fused Triton kernels for LLMs
- [CUDA Graphs in PyTorch](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
