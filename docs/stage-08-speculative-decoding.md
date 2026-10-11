# Stage 8 — Speculative Decoding

> **"2× decode speed at zero quality cost is the closest thing to a free lunch in inference."**

**Status:** ⏳ Pending

---

## What to Learn

- **Draft-target model paradigm** — small model proposes, large model verifies
- **Medusa heads** — multiple draft heads on the target model itself
- **Acceptance rates** — what drives them, how to maximize them
- **N-gram drafting** — zero-cost speculation using recent context

---

## Core Concepts

### The Core Insight

```
Standard decode:
Step 1: run 70B model → 1 token
Step 2: run 70B model → 1 token
...
Cost: N × (70B forward pass)

Speculative decode:
Step 1: run 3B draft × K times → K candidate tokens
         run 70B target ONCE (processes all K tokens in parallel)
         → accept first j tokens where target agrees with draft

If acceptance rate α is high:
  Expected tokens per step = K × α  (e.g., 3 × 0.7 = 2.1 tokens)
  Speedup ≈ 2.1× with identical quality guarantee
```

### Acceptance Criterion (Speculative Sampling)

```
For each proposed token t:
  p = target distribution
  q = draft distribution
  Accept t with probability min(1, p(t)/q(t))
  If rejected: sample from adjusted distribution max(0, p - q)
```

**Output distribution is mathematically identical to the target model.**

### What Determines Acceptance Rate?

- **Draft model quality** — 7B draft for 70B target works better than 1B
- **Task type** — code/structured data → high α (0.7–0.9)
- **Creative generation** → lower α (0.4–0.6)
- **Optimal K** is typically 4–8

### N-gram Drafting

```python
from collections import defaultdict

def build_ngram_table(token_ids: list[int], n: int = 3) -> dict:
    table = defaultdict(list)
    for i in range(len(token_ids) - n):
        context = tuple(token_ids[i:i+n])
        table[context].append(token_ids[i+n])
    return table

def ngram_draft(table: dict, context: list[int], n: int = 3, K: int = 5):
    draft, ctx = [], list(context)
    for _ in range(K):
        key = tuple(ctx[-n:])
        if key in table:
            next_tok = table[key][0]
            draft.append(next_tok)
            ctx.append(next_tok)
        else:
            break
    return draft
```

---

## Practice Project: 3B Draft → 70B Target Pipeline

```bash
# Using vLLM's built-in speculative decoding
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-3B-Instruct \
    --num-speculative-tokens 5 \
    --speculative-draft-tensor-parallel-size 1 \
    --tensor-parallel-size 4
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load models
draft_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda:0"
)
target_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

def speculative_decode(prompt, max_new_tokens=200, K=5, temperature=0.6):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
    generated = input_ids.clone()
    total_tokens, accepted_tokens = 0, 0

    while total_tokens < max_new_tokens:
        with torch.no_grad():
            draft_out = draft_model.generate(
                generated, max_new_tokens=K, do_sample=True,
                temperature=temperature, return_dict_in_generate=True, output_scores=True
            )
        draft_tokens = draft_out.sequences[:, generated.shape[1]:]
        k_actual = draft_tokens.shape[1]

        verify_input = torch.cat([generated, draft_tokens], dim=1).to("cuda")
        with torch.no_grad():
            target_logits = target_model(verify_input).logits
        target_logits_draft = target_logits[:, generated.shape[1]-1:-1, :]

        n_accepted = 0
        for i in range(k_actual):
            p = torch.softmax(target_logits_draft[:, i, :] / temperature, dim=-1)
            q_token = draft_tokens[:, i]
            if torch.rand(1).item() < min(1.0, (p[0, q_token] / 1e-8).item()):
                n_accepted += 1
            else:
                break

        accepted = draft_tokens[:, :n_accepted]
        bonus_pos = generated.shape[1] + n_accepted - 1
        bonus_logits = target_logits[:, bonus_pos, :] / temperature
        bonus_token = torch.multinomial(torch.softmax(bonus_logits, dim=-1), 1)
        generated = torch.cat([generated, accepted.to(generated.device),
                               bonus_token.to(generated.device)], dim=1)
        accepted_tokens += n_accepted
        total_tokens += n_accepted + 1

        if tokenizer.eos_token_id in generated[0, -k_actual:].tolist():
            break

    text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    return text, accepted_tokens / max(total_tokens, 1)

prompt = "Explain the transformer attention mechanism in detail:"
for K in [1, 3, 5, 7, 10]:
    start = time.perf_counter()
    text, alpha = speculative_decode(prompt, max_new_tokens=100, K=K)
    elapsed = time.perf_counter() - start
    print(f"K={K}: α={alpha:.2f} | {100/elapsed:.1f} tok/s")
```

---

## Results Log

| Method | K | Acceptance Rate α | Speedup | Quality (PPL) |
|--------|---|-----------------|---------|---------------|
| Baseline (70B only) | — | — | 1.0× | — |
| Spec (3B draft) | 3 | — | — | identical |
| Spec (3B draft) | 5 | — | — | identical |
| Spec (3B draft) | 7 | — | — | identical |
| N-gram draft | 5 | — | — | identical |

---

## Resources

- [Speculative Decoding Paper (DeepMind)](https://arxiv.org/abs/2302.01318)
- [Speculative Sampling Paper (Google)](https://arxiv.org/abs/2211.17192)
- [Medusa: Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [vLLM Speculative Decoding Docs](https://docs.vllm.ai/en/latest/models/spec_decode.html)
