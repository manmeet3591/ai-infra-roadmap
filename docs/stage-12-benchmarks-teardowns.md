# Stage 12 — Public Benchmarks & Teardowns

> **"Public proof of hardware mastery gets you hired instantly by top AI labs."**

**Status:** ⏳ Pending

---

## What to Learn

- **Reproducible methodology** — configs, seeds, hardware spec, software versions
- **Latency/throughput Pareto curves** — the fundamental tradeoff visualization
- **Cost-per-token analysis** — translating benchmarks into business metrics
- **Publishing a teardown** — structure, what to include, where to post

---

## Core Concepts

### The Latency/Throughput Tradeoff

```
Throughput (tok/s)
      │
  ────┼────────────────────────────────────────── Peak throughput
      │                                  ╱
      │                            ╱────
      │                      ╱────
      │                ╱────
      │─────────────────────────────────────────────── TTFT (ms)
      min                                   max

Each point = a (batch_size, num_scheduler_steps, ...) config.
Goal: find the Pareto frontier for YOUR workload mix.
```

### Cost-Per-Token Analysis

```python
def cost_per_1k_tokens(
    gpu_cost_per_hour: float,
    num_gpus: int,
    tokens_per_second: float,
) -> float:
    cost_per_second = gpu_cost_per_hour * num_gpus / 3600
    cost_per_token = cost_per_second / tokens_per_second
    return cost_per_token * 1000

# 70B on 4×H100 at $3/hr each
print(cost_per_1k_tokens(3.00, 4, 2500))  # ~$0.0048/1K tokens
```

### Benchmark Methodology

```yaml
hardware:
  gpu: "NVIDIA H100 SXM5 80GB"
  gpu_count: 4
  interconnect: "NVLink 4.0"

software:
  vllm_version: "0.6.x"
  cuda_version: "12.4"
  pytorch_version: "2.4.0"

model:
  name: "meta-llama/Llama-3.1-70B-Instruct"
  precision: "bfloat16"
  tensor_parallel: 4

workload:
  input_length_distribution: "sharegpt"
  output_length_mean: 256
  request_rates: [1, 2, 5, 10, 20, 50, 100]
  num_requests: 1000
  random_seed: 42
```

---

## Practice Project: vLLM vs SGLang vs TensorRT-LLM Teardown

```python
import asyncio
import aiohttp
import time
import json
import statistics
from dataclasses import dataclass, field

@dataclass
class BenchmarkResult:
    engine: str
    model: str
    hardware: str
    request_rate: float
    total_requests: int
    successful_requests: int
    total_time_s: float
    ttft_ms: list = field(default_factory=list)
    itl_ms: list = field(default_factory=list)
    output_tokens: list = field(default_factory=list)

    @property
    def throughput_tok_s(self): return sum(self.output_tokens) / self.total_time_s
    @property
    def ttft_p50(self): return statistics.median(self.ttft_ms)
    @property
    def ttft_p99(self): return sorted(self.ttft_ms)[int(len(self.ttft_ms)*0.99)]
    @property
    def itl_p50(self): return statistics.median(self.itl_ms) if self.itl_ms else 0

    def to_dict(self):
        return {
            "engine": self.engine,
            "request_rate": self.request_rate,
            "throughput_tok_s": self.throughput_tok_s,
            "ttft_p50_ms": self.ttft_p50,
            "ttft_p99_ms": self.ttft_p99,
        }

async def benchmark_engine(engine_name, base_url, model, prompts,
                            request_rate, max_output_tokens=256):
    result = BenchmarkResult(engine=engine_name, model=model, hardware="4×H100 SXM5",
                              request_rate=request_rate, total_requests=len(prompts),
                              successful_requests=0, total_time_s=0)

    async def send_request(session, prompt):
        ttft, tokens = None, []
        start = time.perf_counter()
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_output_tokens, "stream": True}
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
            async for line in resp.content:
                line = line.decode().strip()
                if not line.startswith("data: "): continue
                data = line[6:]
                if data == "[DONE]": break
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    tokens.append(content)
        itl = ((time.perf_counter() - start) * 1000 - (ttft or 0)) / max(len(tokens) - 1, 1)
        return ttft, itl, len(tokens)

    interval = 1.0 / request_rate
    start_time = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            await asyncio.sleep(max(0, start_time + i * interval - time.perf_counter()))
            tasks.append(asyncio.create_task(send_request(session, prompt)))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    result.total_time_s = time.perf_counter() - start_time
    for r in results:
        if not isinstance(r, Exception) and r[0]:
            result.ttft_ms.append(r[0])
            result.itl_ms.append(r[1])
            result.output_tokens.append(r[2])
            result.successful_requests += 1
    return result
```

### Publishing Your Teardown

Structure:
1. **Hardware & Software** — full specs table
2. **Methodology** — workload, metrics, request rates swept
3. **Results** — Pareto curve, cost/1K tokens, memory utilization
4. **Conclusions** — what surprised you, what you'd use for which workload
5. **Config files & code** — link to this repo

Where to publish:
- This GitHub Pages site (manmeet3591.github.io/ai-infra-roadmap)
- Twitter/X thread with Pareto curve as lead image
- LinkedIn article
- Hugging Face blog (apply via hf.co/blog)

---

## Final Results Table

| Engine | Throughput (tok/s) | TTFT P99 (ms) | Cost/1K tokens | Config |
|--------|--------------------|--------------|----------------|--------|
| vLLM (FP16) | — | — | — | TP=4 |
| vLLM (FP8) | — | — | — | TP=4 |
| SGLang | — | — | — | default |
| TensorRT-LLM | — | — | — | INT8 |

---

## Resources

- [vLLM Benchmarking Scripts](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- [SGLang Benchmark](https://github.com/sgl-project/sglang/tree/main/benchmark)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [How to Write an AI Benchmark (Chip Huyen)](https://huyenchip.com/2023/01/08/ai-benchmarks.html)
