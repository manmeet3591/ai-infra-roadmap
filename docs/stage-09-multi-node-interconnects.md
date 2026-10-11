# Stage 9 — Multi-Node & Hardware Interconnects

> **"Network latency is the new GPU bottleneck. Disaggregating prefill and decode is the 2026 meta."**

**Status:** ⏳ Pending

---

## What to Learn

- **NCCL** — NVIDIA Collective Communications Library
- **RDMA / InfiniBand** — remote direct memory access, bypassing CPU
- **NVLink** — high-speed GPU-to-GPU interconnect within a node
- **Disaggregated Prefill/Decode** — separate fleets for each phase

---

## Core Concepts

### Interconnect Hierarchy

```
Within GPU:
  Registers → Shared Mem → L2 Cache     (TB/s range)
  
Within node (NVLink):
  GPU 0 ↔ GPU 1 ↔ ... ↔ GPU 8          900 GB/s bidirectional (H100 NVSwitch)
  
Between nodes (InfiniBand):
  Node A ↔ IB Switch ↔ Node B           400 Gb/s (~50 GB/s HDR)
  Node A ↔ IB Switch ↔ Node B           800 Gb/s (~100 GB/s NDR)

Compare:
  NVLink 4.0:    900 GB/s               within-node fast
  IB HDR:         50 GB/s               18× slower!
```

!!! warning "The cross-node penalty"
    TP requires AllReduce per transformer layer.
    Within a node (NVLink): ~0.07ms per 64MB AllReduce
    Across nodes (IB): ~1.3ms per 64MB AllReduce
    **Solution: use PP across nodes, TP within nodes.**

### NCCL Collective Operations

```python
import torch
import torch.distributed as dist

dist.init_process_group("nccl")

# AllReduce: sum across all GPUs, result on all
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# AllGather: each GPU contributes, all get full concat
output_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
dist.all_gather(output_list, tensor)

# AlltoAll: used in MoE expert parallelism
dist.all_to_all(output_list, input_list)
```

### RDMA & InfiniBand

RDMA: network operations bypass CPU entirely.
- NIC transfers data directly between remote memory regions
- Zero CPU involvement during transfer → <2µs latency

```bash
ibstat  # Check IB link speed
ib_write_bw -d mlx5_0 --report_gbits  # test BW
```

### Disaggregated Prefill/Decode

```
Disaggregated:
┌─────────────────────────┐      ┌────────────────────────────┐
│  PREFILL FLEET          │      │  DECODE FLEET              │
│  Compute-optimized      │─KV─►│  Memory-bandwidth tuned    │
│  Large batch prefill    │cache │  Continuous decode         │
└─────────────────────────┘ xfer └────────────────────────────┘
```

---

## Practice Project: Multi-Node Cluster, Profile TP Latency

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500

torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         comm_benchmark.py
```

```python
import torch
import torch.distributed as dist
import time

def benchmark_collective(op_fn, tensor, n_iters=200):
    for _ in range(20):
        op_fn(tensor.clone())
    torch.cuda.synchronize()

    latencies = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(n_iters):
        t = tensor.clone()
        start_evt.record()
        op_fn(t)
        end_evt.record()
        torch.cuda.synchronize()
        latencies.append(start_evt.elapsed_time(end_evt))

    mean_ms = sum(latencies) / len(latencies)
    nbytes = tensor.numel() * tensor.element_size()
    bw = nbytes * 2 / (mean_ms / 1000) / 1e9
    return mean_ms, bw

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = rank % 8
    torch.cuda.set_device(local_rank)

    for size_mb in [1, 4, 16, 64, 128, 256, 512]:
        numel = size_mb * 1024 * 1024 // 2
        tensor = torch.randn(numel, device=f"cuda:{local_rank}", dtype=torch.float16)
        ms, bw = benchmark_collective(lambda t: dist.all_reduce(t), tensor)
        if rank == 0:
            print(f"{size_mb:4d} MB | {ms:.3f} ms | {bw:.1f} GB/s")

if __name__ == "__main__":
    main()
```

---

## Results Log

| Config | Tensor Size | AllReduce (ms) | Effective BW (GB/s) |
|--------|------------|----------------|---------------------|
| 8× H100 NVLink (intra-node) | 64 MB | — | — |
| 2×8 H100 IB HDR (inter-node) | 64 MB | — | — |
| 2×8 H100 IB NDR (inter-node) | 64 MB | — | — |

---

## Resources

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Disaggregated Prefill/Decode (Splitwise)](https://arxiv.org/abs/2311.18677)
- [Mooncake: Disaggregated KV Cache](https://arxiv.org/abs/2407.00079)
- [RDMA Programming Guide](https://www.rdmamojo.com/)
