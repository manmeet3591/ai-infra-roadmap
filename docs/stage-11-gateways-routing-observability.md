# Stage 11 — AI Gateways, Routing & Observability

> **"Routing protects your margins and DCGM metrics tell you when your GPUs are silently throttling."**

**Status:** ⏳ Pending

---

## What to Learn

- **TTFT / ITL SLOs** — defining and monitoring latency service level objectives
- **Semantic routing** — directing requests by content complexity or intent
- **DCGM metrics** — GPU health, power, temperature, throttle events
- **OpenTelemetry for LLMs** — tracing, metrics, logs across the inference stack

---

## Core Concepts

### Key Latency Metrics

| Metric | Definition | Typical SLO |
|--------|-----------|-------------|
| TTFT | Time-To-First-Token | < 500ms (interactive) |
| ITL | Inter-Token Latency (decode) | < 50ms/token |
| E2EL | End-to-End Latency | < 10s (long-form) |
| Throughput | Tokens generated / second | maximize |

```python
import time
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1")
start = time.perf_counter()
first_token_time = None

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain transformers."}],
    stream=True
)

for chunk in stream:
    if first_token_time is None and chunk.choices[0].delta.content:
        first_token_time = time.perf_counter()
        print(f"TTFT: {(first_token_time - start)*1000:.0f}ms")
    print(chunk.choices[0].delta.content or "", end="", flush=True)

print(f"\nE2E: {(time.perf_counter() - start)*1000:.0f}ms")
```

### Semantic Routing by Complexity

```python
from fastapi import FastAPI, Request
import tiktoken, httpx

app = FastAPI()
ROUTES = {
    "fast": "http://localhost:8001",   # quantized 8B
    "smart": "http://localhost:8002",  # 70B
}
COMPLEXITY_THRESHOLD = 150

def estimate_complexity(messages: list) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    text = " ".join(m.get("content", "") for m in messages)
    n_tokens = len(enc.encode(text))
    reasoning_keywords = ["analyze", "compare", "evaluate", "step by step",
                          "reason", "explain why", "proof", "derive"]
    keyword_boost = sum(5 for kw in reasoning_keywords if kw in text.lower())
    return n_tokens + keyword_boost

@app.post("/v1/chat/completions")
async def route_request(request: Request):
    body = await request.json()
    complexity = estimate_complexity(body.get("messages", []))
    target = ROUTES["smart" if complexity > COMPLEXITY_THRESHOLD else "fast"]
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{target}/v1/chat/completions", json=body)
    return resp.json()
```

### DCGM Metrics — GPU Health Monitoring

```bash
apt-get install -y datacenter-gpu-manager
systemctl start nvidia-dcgm
dcgmi dmon -e 203,204,155,156,100
dcgm-exporter &  # Prometheus metrics at :9400/metrics
```

| DCGM Field | Metric | Alert Threshold |
|-----------|--------|----------------|
| 155 | GPU Utilization % | < 70% during requests |
| 156 | Memory Utilization % | > 95% |
| 203 | Power Draw (W) | > 95% TDP |
| 204 | GPU Temperature (°C) | > 83°C |
| 230 | Throttle Reasons | any bit set |
| 1001 | XID Errors | any |

### OpenTelemetry for LLM Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import time

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://tempo:4317")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-gateway")
FastAPIInstrumentor.instrument_app(app)

@app.post("/v1/chat/completions")
async def inference(request: Request):
    body = await request.json()
    with tracer.start_as_current_span("llm.inference") as span:
        span.set_attribute("llm.model", body.get("model", "unknown"))
        start = time.perf_counter()
        result = await call_backend(body)
        span.set_attribute("llm.latency_ms", (time.perf_counter() - start) * 1000)
    return result
```

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm-server:8000']
  - job_name: 'dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']
```

---

## Practice Project: Complexity-Routing Gateway

```bash
uvicorn gateway:app --host 0.0.0.0 --port 9000

# Simple request → fast model
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "model": "auto", "max_tokens": 50}'

# Complex request → 70B
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Analyze the computational complexity of attention mechanisms vs state space models, deriving FLOPs per token for each."}], "model": "auto", "max_tokens": 500}'
```

---

## Results Log

| Request Type | Routed To | Avg TTFT | Cost/1K tok | Quality (1-5) |
|-------------|----------|---------|------------|---------------|
| Simple Q&A | 8B quant | — | — | — |
| Complex reasoning | 70B | — | — | — |
| Code generation | 70B | — | — | — |

---

## Resources

- [vLLM Metrics](https://docs.vllm.ai/en/latest/serving/metrics.html)
- [DCGM Exporter for Prometheus](https://github.com/NVIDIA/dcgm-exporter)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [LiteLLM Router](https://docs.litellm.ai/docs/routing)
