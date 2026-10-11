# Stage 4 — KV-Cache & Memory Optimization

> **"Reused cache is free speed. It can slash Time-To-First-Token by 80%."**

**Status:** ⏳ Pending

---

## What to Learn

- **Prefix caching** — reusing KV blocks for shared prompt prefixes
- **KV quantization** — reducing cache precision to INT8/INT4
- **CPU offloading** — spilling KV cache to CPU DRAM between requests
- **Multi-turn cache reuse** — sharing cache across conversation turns

---

## Core Concepts

### Prefix Caching

```
Without prefix caching:
Request 1: [system prompt = 500 tok][user query A = 20 tok] → compute all 520 tokens
Request 2: [system prompt = 500 tok][user query B = 20 tok] → compute all 520 tokens again
Cost: 3 × 520 = 1560 token-prefill-compute

With prefix caching:
Request 1: compute 520 tokens, cache KV for first 500
Request 2: cache HIT → compute only 20 new tokens
Cost: 520 + 20 + 20 = 560 token-prefill-compute  (64% reduction)
```

!!! success "TTFT reduction"
    For a 500-token system prompt, prefill takes ~50ms. With a cache hit,
    TTFT drops to ~5ms. That's an **80–90% TTFT reduction** for all requests after the first.

### Hash-Based Prefix Matching

```python
import hashlib

def compute_block_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(bytes(token_ids)).hexdigest()

class PrefixCache:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.cache: dict[str, object] = {}

    def lookup(self, token_ids: list[int]) -> int:
        cached = 0
        for i in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            block = token_ids[i:i + self.block_size]
            h = compute_block_hash(block)
            if h not in self.cache:
                break
            cached += self.block_size
        return cached
```

### KV Quantization

| Format | Bytes/element | Memory vs FP16 | Quality Impact |
|--------|-------------|----------------|---------------|
| FP16 | 2 | 1× | baseline |
| INT8 | 1 | 0.5× | negligible (<0.1 ppl) |
| INT4 | 0.5 | 0.25× | small (0.1–0.5 ppl) |
| FP8 | 1 | 0.5× | negligible |

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --kv-cache-dtype fp8
```

### CPU Offloading

```
GPU VRAM (80 GB H100):  [Model weights: 70 GB][Hot KV blocks: 10 GB]
                                                          ↕ swap
CPU DRAM (384 GB):      [Cold KV blocks for idle requests: 100+ GB]
```

Tradeoff: PCIe bandwidth (~64 GB/s) means swapping introduces ~1–5ms latency.

---

## Practice Project: KV-Cache Routing Proxy

```python
from fastapi import FastAPI, Request
import httpx
import hashlib
from typing import Dict

app = FastAPI()

REPLICAS = [
    "http://replica-a:8000",
    "http://replica-b:8001",
    "http://replica-c:8002",
]

prefix_map: Dict[str, str] = {}

def extract_system_prompt(messages: list) -> str:
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    return ""

def route_request(system_prompt: str) -> str:
    h = hashlib.sha256(system_prompt.encode()).hexdigest()[:8]
    if h in prefix_map:
        return prefix_map[h]
    replica = REPLICAS[len(prefix_map) % len(REPLICAS)]
    prefix_map[h] = replica
    return replica

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    body = await request.json()
    system_prompt = extract_system_prompt(body.get("messages", []))
    target = route_request(system_prompt)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{target}/v1/chat/completions", json=body)
    return resp.json()
```

### Measuring Cache Hit Rate

```python
import requests, time

system_prompt = "You are an expert Python engineer. " * 50  # 250 tokens

def timed_request(user_msg: str) -> float:
    start = time.perf_counter()
    requests.post("http://localhost:9000/v1/chat/completions", json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": 50
    })
    return (time.perf_counter() - start) * 1000

ttft_cold = timed_request("Explain list comprehensions.")
print(f"Cold TTFT: {ttft_cold:.0f}ms")

for i in range(5):
    ttft_warm = timed_request(f"Question number {i}")
    print(f"Warm TTFT: {ttft_warm:.0f}ms")
```

---

## Results Log

| Scenario | Prefix Length | Cold TTFT | Warm TTFT | Cache Hit Reduction |
|----------|-------------|----------|----------|-------------------|
| Short system prompt | 100 tok | — | — | — |
| Long system prompt | 500 tok | — | — | — |
| KV INT8 quantization | — | — | — | — |
| CPU offload | — | — | — | — |

---

## Resources

- [Automatic Prefix Caching in vLLM](https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html)
- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [KV Cache Compression Survey](https://arxiv.org/abs/2408.03130)
