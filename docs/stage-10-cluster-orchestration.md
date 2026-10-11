# Stage 10 — Cluster Orchestration & GPU Scheduling

> **"Idle H100s burn $3+/hr. FinOps and scheduling are now core infra responsibilities."**

**Status:** ⏳ Pending

---

## What to Learn

- **Kubernetes GPU operators** — NVIDIA device plugin, resource limits
- **Ray** — Python-native distributed computing for ML
- **Slurm** — HPC-style batch scheduling
- **MIG (Multi-Instance GPU)** — partitioning a single GPU into isolated instances
- **KEDA** — Kubernetes-based event-driven autoscaling

---

## Core Concepts

### Kubernetes GPU Resources

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llm-server
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    resources:
      limits:
        nvidia.com/gpu: "4"
        memory: "256Gi"
      requests:
        nvidia.com/gpu: "4"
    command: ["python", "-m", "vllm.entrypoints.openai.api_server",
              "--model", "meta-llama/Llama-3.1-70B-Instruct",
              "--tensor-parallel-size", "4"]
```

### MIG (Multi-Instance GPU) Partitioning

An H100 80GB can be partitioned into up to 7 independent MIG instances.

```bash
nvidia-smi -i 0 -mig 1
nvidia-smi mig -cgi 3g.40gb,3g.40gb -C   # Two 40GB instances
nvidia-smi mig -cgi 2g.20gb,2g.20gb,2g.20gb -C  # Three 20GB
nvidia-smi -L
```

### Ray Serve for LLM Serving

```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "engine_kwargs": {"tensor_parallel_size": 1, "max_model_len": 8192}
    },
    deployment_config={"num_replicas": 2, "ray_actor_options": {"num_gpus": 1}}
)
app = build_openai_app({"llm": llm_config})
serve.run(app)
```

### KEDA — Queue-Based Autoscaling

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaledobject
spec:
  scaleTargetRef:
    name: vllm-deployment
  minReplicaCount: 1
  maxReplicaCount: 10
  cooldownPeriod: 60
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: vllm_pending_requests
      query: sum(vllm_num_requests_waiting)
      threshold: "5"
```

---

## Practice Project: Queue-Based GPU Autoscaler

```python
import time
import boto3
import redis
from dataclasses import dataclass

@dataclass
class ScalerConfig:
    queue_name: str = "inference-queue"
    min_workers: int = 1
    max_workers: int = 8
    requests_per_worker: int = 5
    scale_up_threshold: int = 10
    scale_down_delay_s: int = 120
    check_interval_s: int = 15

class GPUAutoscaler:
    def __init__(self, config: ScalerConfig):
        self.config = config
        self.redis = redis.Redis(host='localhost', port=6379)
        self.ec2 = boto3.client('ec2', region_name='us-east-1')
        self.active_instances: list[str] = []
        self.idle_since: dict[str, float] = {}

    def get_queue_depth(self) -> int:
        return self.redis.llen(self.config.queue_name)

    def desired_workers(self, queue_depth: int) -> int:
        needed = max(self.config.min_workers,
                     (queue_depth + self.config.requests_per_worker - 1)
                     // self.config.requests_per_worker)
        return min(needed, self.config.max_workers)

    def scale_up(self):
        response = self.ec2.run_instances(
            ImageId='ami-xxxxx',
            InstanceType='p3.2xlarge',
            InstanceMarketOptions={'MarketType': 'spot',
                                   'SpotOptions': {'SpotInstanceType': 'one-time'}},
            MinCount=1, MaxCount=1,
            UserData="""#!/bin/bash
                python -m vllm.entrypoints.openai.api_server \\
                  --model meta-llama/Llama-3.1-8B-Instruct --port 8000 &
            """,
            TagSpecifications=[{'ResourceType': 'instance',
                                'Tags': [{'Key': 'role', 'Value': 'llm-worker'}]}]
        )
        instance_id = response['Instances'][0]['InstanceId']
        self.active_instances.append(instance_id)
        print(f"Launched {instance_id}")

    def scale_down(self, instance_id: str):
        self.ec2.terminate_instances(InstanceIds=[instance_id])
        self.active_instances.remove(instance_id)
        print(f"Terminated {instance_id}")

    def run(self):
        while True:
            depth = self.get_queue_depth()
            current = len(self.active_instances)
            desired = self.desired_workers(depth)
            if desired > current:
                for _ in range(desired - current):
                    self.scale_up()
            elif desired < current and current > self.config.min_workers:
                idle = self.active_instances[-1]
                if idle not in self.idle_since:
                    self.idle_since[idle] = time.time()
                elif time.time() - self.idle_since[idle] > self.config.scale_down_delay_s:
                    self.scale_down(idle)
                    del self.idle_since[idle]
            print(f"Queue: {depth} | Workers: {current} → {desired}")
            time.sleep(self.config.check_interval_s)

if __name__ == "__main__":
    GPUAutoscaler(ScalerConfig()).run()
```

---

## Results Log

| Setup | Idle GPU hours saved | P95 Queue Wait | Cost/1K requests |
|-------|---------------------|----------------|------------------|
| Always-on (baseline) | — | — | — |
| Autoscaled (spot) | — | — | — |
| MIG-partitioned | — | — | — |

---

## Resources

- [NVIDIA GPU Operator for Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)
- [KEDA Documentation](https://keda.sh/docs/)
- [Ray Serve LLM](https://docs.ray.io/en/latest/serve/llm/)
- [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [Slurm GPU Scheduling](https://slurm.schedmd.com/gres.html)
