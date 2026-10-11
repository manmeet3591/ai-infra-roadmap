# Stage 2 — Transformer Inference Physics

> **"LLM inference is almost always memory-bound, not compute-bound."**

**Status:** ⏳ Pending

---

## What to Learn

- **Prefill vs Decode phases** — fundamentally different compute profiles
- **Arithmetic intensity** — FLOPs per byte of memory moved (the roofline model)
- **Memory bandwidth bottlenecks** — why weights dominate at small batch sizes
- **KV-cache math** — memory footprint formula, growth per token

---

## Core Concepts

### The Two Phases of Inference

```
PREFILL                          DECODE
───────────────────────         ─────────────────────────
Input: full prompt tokens        Input: 1 new token per step
Output: first token + KV cache   Output: 1 token per step

Compute profile: HIGH            Compute profile: LOW
(matrix × matrix, parallel)      (matrix × vector, sequential)

Bound by: COMPUTE                Bound by: MEMORY BANDWIDTH
```

!!! info "Why this asymmetry matters"
    During prefill you're doing a large GEMM (compute-bound).
    During decode you're loading the entire model weight matrix just to multiply
    it by a single vector. You use ~1% of your compute capacity and waste 99%.
    Every optimization for decode is about hiding or reducing memory bandwidth cost.

### Arithmetic Intensity & the Roofline Model

**Arithmetic intensity** = FLOPs / bytes of memory traffic

For a 70B parameter model (BF16) at batch_size=1:
- Weights: 140 GB
- FLOPs per token (decode): ~140 × 10⁹ multiply-adds = 280 GFLOPs
- Arithmetic intensity: 280e9 / 140e9 = **~2 FLOPs/byte**
- H100 ridge point: ~300 FLOPs/byte

**Result: 2 << 300 → deeply memory-bound.**

### KV-Cache Memory Math

```
KV cache size = 2 × num_layers × num_heads × head_dim × seq_len × bytes_per_element

Example (Llama-3 70B, FP16, seq_len=8192):
= 2 × 80 × 8 × 128 × 8192 × 2 bytes
≈ 42 GB  (!)
```

### Memory Bandwidth vs Compute Bound Regimes

| Batch Size | Phase | Bottleneck |
|-----------|-------|----------|
| 1 | Decode | Memory bandwidth (weight loading) |
| 1 | Prefill | Compute (GEMM) |
| Large (≥32) | Decode | Can become compute-bound |
| Large | Prefill | Always compute-bound |

---

## Practice Project: Profile HuggingFace with Nsight

**Goal:** Find the exact memory bottleneck during token generation on a real model.

```bash
pip install transformers accelerate
nsys profile --trace=cuda,nvtx --output=llm_profile python inference_profile.py
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nvtx

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda"
)

prompt = "Explain the attention mechanism in transformers:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with nvtx.annotate("PREFILL", color="green"):
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values

with nvtx.annotate("DECODE", color="red"):
    for step in range(50):
        with nvtx.annotate(f"decode_step_{step}"):
            with torch.no_grad():
                next_token_logits = outputs.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_kv,
                    use_cache=True
                )
                past_kv = outputs.past_key_values
```

```python
def compute_decode_arithmetic_intensity(model):
    total_params = sum(p.numel() for p in model.parameters())
    bytes_moved = total_params * 2
    flops = total_params * 2
    return flops / bytes_moved
# Should be ~1-2, confirming memory-bound operation
```

---

## Results Log

| Model | Phase | SM Util % | Mem BW Util % | TTFT (ms) | Decode ms/tok |
|-------|-------|-----------|--------------|-----------|---------------|
| — | Prefill | — | — | — | — |
| — | Decode | — | — | — | — |

---

## Resources

- [Dissecting Batching Effects in GPT Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)
