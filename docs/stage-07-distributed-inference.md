# Stage 7 — Distributed Inference & Parallelism

> **"Single-GPU inference is dead for frontier models. You must master the shard."**

**Status:** ⏳ Pending

---

## What to Learn

- **Tensor Parallelism (TP)** — split individual weight matrices across GPUs
- **Pipeline Parallelism (PP)** — split model layers across GPUs
- **Expert Parallelism (EP)** — distribute MoE experts across GPUs
- **Communication patterns** — AllReduce, AllGather, scatter, pipeline bubbles

---

## Core Concepts

### Tensor Parallelism (Megatron-style)

```
Column Parallel (split A vertically across N GPUs):
GPU 0: Y₀ = GeLU(X · A₀)   [A₀: d×(4d/N)]
GPU 1: Y₁ = GeLU(X · A₁)   [A₁: d×(4d/N)]
→ AllGather to get full Y

Row Parallel (second linear layer):
GPU 0: Z₀ = Y₀ · B₀
GPU 1: Z₁ = Y₁ · B₁
→ AllReduce Z = Σ Zᵢ
```

### Communication Volume for TP

At batch=1, seq=2048, hidden=8192 (70B), FP16:
- Per AllReduce: 2 × 1 × 2048 × 8192 × 2 = **67 MB**
- 80 layers × 2 AllReduces = **160 AllReduces per forward pass**
- At 900 GB/s NVLink: 67 MB / 900 GB/s = **0.07 ms per AllReduce**
- Total comm overhead: **~11 ms** — significant!

### Pipeline Parallelism

```
Layers 0–19  → GPU 0
Layers 20–39 → GPU 1
Layers 40–59 → GPU 2
Layers 60–79 → GPU 3

Bubble fraction = (PP_degree - 1) / num_micro_batches
```

### Expert Parallelism (MoE)

```
Token routing: Input → Router → top-K experts selected
GPU 0: Expert 0, 4, 8, ...
GPU 1: Expert 1, 5, 9, ...
→ AlltoAll: gather outputs back to original token positions
```

---

## Practice Project: Shard a 405B Model, Measure Comm Overhead

```bash
# 8×H100 on single node (NVLink)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 4096

# Multi-node: TP=8 PP=2
# Node 0 (head)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --dtype bfloat16
```

```python
import torch
import torch.distributed as dist
import time

def benchmark_allreduce(tensor_size_mb: float, n_iters: int = 100):
    size = int(tensor_size_mb * 1e6 / 2)  # FP16 = 2 bytes
    tensor = torch.randn(size, device=f'cuda:{dist.get_rank()}', dtype=torch.float16)

    for _ in range(10):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000
    bw = tensor_size_mb * 2 / elapsed * 1000
    return elapsed, bw

if __name__ == "__main__":
    dist.init_process_group("nccl")
    for size in [1, 8, 32, 64, 128, 256]:
        ms, bw = benchmark_allreduce(size)
        if dist.get_rank() == 0:
            print(f"{size:4d} MB AllReduce: {ms:.2f} ms | {bw:.0f} GB/s")
```

---

## Results Log

| Config | Model | Throughput (tok/s) | Comm Overhead (ms/layer) | HW |
|--------|-------|--------------------|--------------------------|-----|
| TP=1 | 8B | — | — | 1×H100 |
| TP=4 | 70B | — | — | 4×H100 NVLink |
| TP=8 | 405B | — | — | 8×H100 NVLink |
| TP=8 PP=2 | 405B | — | — | 2×8×H100 IB |

---

## Resources

- [Megatron-LM Parallelism](https://arxiv.org/abs/2104.04473)
- [vLLM Distributed Serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [MoE Expert Parallelism (Mixtral)](https://arxiv.org/abs/2401.04088)
