# AI Infrastructure Engineering

> **"Wrappers are a commodity. AI Infrastructure is the physics of scale."**

This is a living engineering journal tracking hands-on progress through the full AI infrastructure stack — from CUDA memory hierarchies to multi-node serving clusters. Each stage pairs conceptual depth with a concrete practice project and benchmarks.

---

## The 12-Stage Roadmap

| Stage | Topic | Status |
|-------|-------|--------|
| [1](stage-01-systems-gpu-foundations.md) | Systems & GPU Foundations | 🔄 In Progress |
| [2](stage-02-transformer-inference-physics.md) | Transformer Inference Physics | ⏳ Pending |
| [3](stage-03-serving-engines-batching.md) | Modern Serving Engines & Batching | ⏳ Pending |
| [4](stage-04-kv-cache-memory.md) | KV-Cache & Memory Optimization | ⏳ Pending |
| [5](stage-05-quantization-compression.md) | Quantization & Compression | ⏳ Pending |
| [6](stage-06-kernel-engineering.md) | Kernel-Level Engineering | ⏳ Pending |
| [7](stage-07-distributed-inference.md) | Distributed Inference & Parallelism | ⏳ Pending |
| [8](stage-08-speculative-decoding.md) | Speculative Decoding | ⏳ Pending |
| [9](stage-09-multi-node-interconnects.md) | Multi-Node & Hardware Interconnects | ⏳ Pending |
| [10](stage-10-cluster-orchestration.md) | Cluster Orchestration & GPU Scheduling | ⏳ Pending |
| [11](stage-11-gateways-routing-observability.md) | AI Gateways, Routing & Observability | ⏳ Pending |
| [12](stage-12-benchmarks-teardowns.md) | Public Benchmarks & Teardowns | ⏳ Pending |

---

## Why This Exists

LLM inference is almost always **memory-bound, not compute-bound**. Most engineers optimizing AI systems are operating at the wrong level of abstraction — tuning Python when the bottleneck is on-chip bandwidth, or scaling horizontally when the real fix is kernel fusion.

This journal is built on the conviction that you cannot optimize what you don't understand at the silicon level. Each stage builds the mental model *and* produces a concrete artifact — a profiled kernel, a serving benchmark, a routing proxy — that proves the understanding is real.

---

## Stack & Hardware

!!! info "Environment"
    Hardware and software details will be filled in here as the journey progresses.

---

## Key References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [PagedAttention / vLLM Paper](https://arxiv.org/abs/2309.06180)
- [Triton Language](https://triton-lang.org/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
