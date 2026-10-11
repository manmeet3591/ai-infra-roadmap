# Stage 3 — Modern Serving Engines & Batching

> **"Naive batching leaves 60% of GPU VRAM wasted. PagedAttention fixes this."**

**Status:** ⏳ Pending

---

## What to Learn

- **vLLM** — PagedAttention, continuous batching architecture
- **SGLang** — RadixAttention, structured generation, prefix caching
- **PagedAttention** — paged virtual memory for KV cache
- **Continuous batching** — dynamic request scheduling vs static batching
- **Chunked prefill** — interleaving prefill chunks with decode steps

---

## Core Concepts

### The Problem with Naive Batching

In static batching, you wait until a batch is full, run it, wait again:

```
Static Batching:
Time ──►
[req1 ██████████████████████████████████████]  GPU idle waiting
[req2 ██████████████████████████████████████]
[req3 ██████████████████████████████████████]
──────────────────────────────────────────────
                  BATCH RUNS                   New requests queue
```

Problems:
1. **Head-of-line blocking** — short requests wait for long ones to finish
2. **VRAM fragmentation** — each request pre-allocates its maximum sequence length
3. **GPU underutilization** — GPU idles while batch is being assembled

### Continuous Batching

```
Continuous Batching:
Time ──►
[req1  ████ done ]
[req2  ████████████████ done ]
[req3  ██ done ]
[req4           ████ inserted immediately when req1 done ]
[req5                    ████████ inserted when req3 done ]
──────────────────────────────────────────────────────────
        GPU always working — no idle time
```

### PagedAttention

```
Traditional:
[req1: max_len=2048 ████████████████ (actually uses 500 tokens)]
[req2: max_len=2048 ████████████████ (actually uses 200 tokens)]
Wasted: ~60% of allocated VRAM

PagedAttention:
Physical blocks (e.g., 16 tokens each):
[blk0][blk1][blk2][blk3]...[blkN]
Logical pages (per request):
req1 → [blk0][blk3][blk7]           (non-contiguous, grows on demand)
req2 → [blk1][blk5]                  (non-contiguous, grows on demand)
Wasted: ~4% (just last partial page per request)
```

!!! tip "Why this is transformative"
    PagedAttention lets you serve 3–4× more concurrent requests on the same GPU
    by eliminating KV cache fragmentation.

### Chunked Prefill

```
Without chunked prefill:
[long prefill: 32K tokens ████████████████████████] decode starved

With chunked prefill (chunk_size = 2048):
[prefill chunk 1 ██][decode][prefill chunk 2 ██][decode]...
TTFT for interactive requests stays low while long context is processed
```

---

## Practice Project: Deploy 70B & Tune Chunk Sizes

```bash
pip install vllm

# Single GPU
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --port 8000

# Multi-GPU 70B
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --port 8000
```

```python
import asyncio
import aiohttp
import time

async def send_request(session, prompt, max_tokens=100):
    start = time.perf_counter()
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False
    }
    async with session.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload
    ) as resp:
        result = await resp.json()
        ttft = time.perf_counter() - start
        return ttft, result

async def load_test(num_requests=50):
    short_prompt = "What is 2+2?"
    long_prompt = "Summarize the history of " + "artificial intelligence " * 200

    async with aiohttp.ClientSession() as session:
        all_tasks = (
            [send_request(session, short_prompt) for _ in range(num_requests)] +
            [send_request(session, long_prompt, max_tokens=200) for _ in range(5)]
        )
        results = await asyncio.gather(*all_tasks)

    ttfts = [r[0] for r in results]
    print(f"P50 TTFT: {sorted(ttfts)[len(ttfts)//2]*1000:.0f}ms")
    print(f"P99 TTFT: {sorted(ttfts)[int(len(ttfts)*0.99)]*1000:.0f}ms")

asyncio.run(load_test())
```

---

## Results Log

| Engine | Chunk Size | P50 TTFT | P99 TTFT | Throughput (tok/s) | VRAM Used |
|--------|-----------|---------|---------|------------------|----------|
| vLLM (no chunked prefill) | — | — | — | — | — |
| vLLM | 512 | — | — | — | — |
| vLLM | 2048 | — | — | — | — |
| SGLang | default | — | — | — | — |

---

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [SGLang Paper](https://arxiv.org/abs/2312.07104)
- [Continuous Batching (Orca paper)](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Chunked Prefill in vLLM](https://blog.vllm.ai/2024/09/05/perf-update.html)
