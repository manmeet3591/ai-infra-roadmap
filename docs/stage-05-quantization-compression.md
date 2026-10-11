# Stage 5 — Quantization & Compression

> **"Quantization is the only way to fit frontier models on edge GPUs and protect your margins."**

**Status:** ⏳ Pending

---

## What to Learn

- **FP8 / INT4** — number formats, range vs precision tradeoffs
- **AWQ** — Activation-aware Weight Quantization
- **GPTQ** — post-training quantization with Hessian-based rounding
- **Sparsity** — structured vs unstructured pruning
- **TensorRT-LLM calibration** — end-to-end quantized serving pipeline

---

## Core Concepts

### Number Format Comparison

```
FP32  [sign:1][exp:8][mantissa:23]   Range: ±3.4e38   Precision: ~7 sig digits
FP16  [sign:1][exp:5][mantissa:10]   Range: ±65504    Precision: ~3 sig digits
BF16  [sign:1][exp:8][mantissa:7]    Range: ±3.4e38   Precision: ~2 sig digits
FP8   [sign:1][exp:4][mantissa:3]    Range: ±448      Precision: ~1 sig digit
INT8  [-128 to 127]                   Linear scale      No exponent
INT4  [-8 to 7]                       Linear scale      Very coarse
```

!!! warning "Why INT4 is tricky"
    With only 16 representable values, activations with outliers cause large
    quantization error. AWQ and GPTQ both address this with per-channel or
    per-group scaling.

### VRAM & Throughput Comparison

| Format | Llama 3.1 70B Size | Throughput vs FP16 | Perplexity ↑ |
|--------|------------------|--------------------|------------|
| FP16 | 140 GB | 1× | baseline |
| BF16 | 140 GB | ~1× | baseline |
| FP8 | 70 GB | 1.3–1.6× | +0.01–0.05 |
| INT8 (SmoothQuant) | 70 GB | 1.2–1.4× | +0.05–0.1 |
| INT4 (AWQ) | 35 GB | 1.5–2× | +0.1–0.3 |
| INT4 (GPTQ) | 35 GB | 1.5–2× | +0.1–0.5 |

### AWQ (Activation-Aware Weight Quantization)

Key insight: not all weights are equally important. Weights that correspond
to large activation channels cause outsized quantization error.

```bash
pip install autoawq

python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'meta-llama/Llama-3.1-8B-Instruct'
quant_path = 'llama-3.1-8b-awq'

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

quant_config = {
    'zero_point': True,
    'q_group_size': 128,
    'w_bit': 4,
    'version': 'GEMM'
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
"
```

### GPTQ

```bash
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-70B-GPTQ \
    --quantization gptq \
    --dtype float16
```

### FP8 on H100

H100 natively supports FP8 compute (E4M3 format), enabling ~2× throughput vs FP16.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --kv-cache-dtype fp8
```

---

## Practice Project: FP8 vs FP16 Benchmark

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time, math

def compute_perplexity(model, tokenizer, max_samples=50) -> float:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"][:max_samples]), return_tensors="pt")
    max_length, stride, nlls = 2048, 512, []
    for begin_loc in range(0, min(encodings.input_ids.size(1), 8192), stride):
        end_loc = min(begin_loc + max_length, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss.item())
        if end_loc >= encodings.input_ids.size(1):
            break
    return math.exp(sum(nlls) / len(nlls))

def measure_throughput(model, tokenizer, batch_size=1, seq_len=512, num_new_tokens=128):
    dummy_input = torch.randint(0, 32000, (batch_size, seq_len), device="cuda")
    start = time.perf_counter()
    with torch.no_grad():
        model.generate(dummy_input, max_new_tokens=num_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    return (batch_size * num_new_tokens) / (time.perf_counter() - start)

model_name = "meta-llama/Llama-3.2-1B"
results = {}

for dtype, torch_dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    print(f"\n=== {dtype} ===")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vram = torch.cuda.max_memory_allocated() / 1e9
    ppl = compute_perplexity(model, tokenizer)
    tps = measure_throughput(model, tokenizer)
    results[dtype] = {"vram_gb": vram, "perplexity": ppl, "tokens_per_sec": tps}
    print(f"  VRAM: {vram:.1f} GB | PPL: {ppl:.2f} | Throughput: {tps:.0f} tok/s")
    del model
    torch.cuda.empty_cache()
```

---

## Results Log

| Precision | Model | VRAM (GB) | Perplexity | Perplexity Δ | Throughput (tok/s) |
|-----------|-------|----------|-----------|-------------|------------------|
| FP16 | — | — | — | 0.00 | — |
| FP8 | — | — | — | — | — |
| AWQ INT4 | — | — | — | — | — |
| GPTQ INT4 | — | — | — | — | — |

---

## Resources

- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [TensorRT-LLM Quantization](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)
- [LLM.int8() Paper (bitsandbytes)](https://arxiv.org/abs/2208.07339)
