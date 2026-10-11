# Stage 1 — Systems & GPU Foundations

> **"You cannot optimize what you do not understand at the silicon level."**

**Status:** 🔄 In Progress

---

## What to Learn

- **C++ & Rust** — systems-level memory management, pointer arithmetic, ownership models
- **CUDA Memory Hierarchy** — registers → L1/shared memory → L2 → HBM (global memory)
- **PCIe vs NVLink** — bandwidth, latency, topology implications for multi-GPU systems
- **Thread blocks & warp scheduling** — SIMT execution model, divergence, occupancy

---

## Core Concepts

### CUDA Memory Hierarchy

```
Register File       ~10 TB/s    Per-thread, fastest
─────────────────────────────────────────────────────
Shared Memory / L1  ~20 TB/s    Per-SM, programmer-managed
─────────────────────────────────────────────────────
L2 Cache            ~10 TB/s    On-chip, shared across SMs
─────────────────────────────────────────────────────
HBM (Global DRAM)   ~3.35 TB/s  Off-chip (H100 SXM5)
─────────────────────────────────────────────────────
PCIe (host↔device)  ~64 GB/s    System interconnect
NVLink 4.0          ~900 GB/s   GPU↔GPU (H100)
```

!!! warning "The Bandwidth Cliff"
    Moving data from shared memory to global memory is ~6× slower.
    Moving from GPU to CPU over PCIe is ~50× slower than HBM bandwidth.
    Most inference bottlenecks trace back to unnecessary memory movement.

### Warp Scheduling

- A **warp** is 32 threads executing in lockstep (SIMT)
- An **SM** (Streaming Multiprocessor) on H100 has 128 CUDA cores → runs 4 warps simultaneously
- **Occupancy** = active warps / maximum warps per SM
- High occupancy hides memory latency by switching warps while one waits on a memory fetch

### PCIe vs NVLink

| Interconnect | Bandwidth (bidirectional) | Latency |
|---|---|---|
| PCIe 5.0 x16 | ~128 GB/s | ~1–2 µs |
| NVLink 4.0 | ~900 GB/s | <1 µs |
| NVLink 5.0 (B100) | ~1800 GB/s | <1 µs |

NVLink matters for tensor parallelism — all-reduce operations between GPUs use this fabric.

---

## Practice Project: CUDA Matrix Multiplication

**Goal:** Write a CUDA kernel for matrix multiplication from scratch, then profile it against cuBLAS.

### Naive Kernel

```cuda
__global__ void matmul_naive(float* A, float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Problem:** Each thread reads from global memory K times. For K=4096, that's 4096 uncached global reads per thread. Memory bandwidth is destroyed.

### Tiled Kernel (Shared Memory)

```cuda
#define TILE_SIZE 32

__global__ void matmul_tiled(float* A, float* B, float* C,
                              int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### Benchmarking Template

```python
import torch
import time

def benchmark(fn, *args, warmup=5, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1000  # ms

M, N, K = 4096, 4096, 4096
A = torch.randn(M, K, device='cuda', dtype=torch.float32)
B = torch.randn(K, N, device='cuda', dtype=torch.float32)

cublas_time = benchmark(torch.matmul, A, B)
flops = 2 * M * N * K
tflops = flops / (cublas_time / 1000) / 1e12

print(f"cuBLAS: {cublas_time:.2f} ms | {tflops:.1f} TFLOP/s")
```

---

## Profiling with Nsight Systems

```bash
nsys profile --stats=true python matmul_benchmark.py
ncu --set full python matmul_benchmark.py
```

**Key metrics to check:**
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` — global memory reads
- `sm__warps_active.avg.pct_of_peak_sustained_active` — occupancy
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM utilization

---

## Results Log

| Kernel | Matrix Size | Time (ms) | TFLOP/s | % of cuBLAS |
|--------|------------|-----------|---------|-------------|
| Naive  | 4096×4096  | —         | —       | — |
| Tiled  | 4096×4096  | —         | —       | — |
| cuBLAS | 4096×4096  | —         | —       | 100% |

*Fill in after running benchmarks.*

---

## Resources

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [CUDA by Example (book)](https://developer.nvidia.com/cuda-example)
