---
title: CloudScalerEnv
sdk: docker
app_port: 7860
tags:
  - openenv
  - devops
  - sre
---

# CloudScalerEnv (OpenEnv)

CloudScalerEnv is a production-grade SRE simulation environment where autonomous agents learn to manage microservice infrastructure under realistic stress conditions. The environment models the critical decision-making workflows that SRE teams perform daily: scaling services to handle traffic spikes, restarting unhealthy services, managing cascading failures, and maintaining tight cost budgets while preventing service outages.

## Quickstart Client

Run one end-to-end client flow in under a minute:

```bash
python examples/manual_episode.py
```

Async variant:

```bash
python examples/async_client_episode.py
```

## Motivation: Why This Matters

Modern cloud-native systems require 24/7 SRE oversight. Production SRE teams spend significant time:
- **Detecting abnormal metrics** (CPU spikes, memory leaks) and responding within minutes
- **Orchestrating multi-step remediation** (e.g., "scale this service, monitor for cascade effects, then restart if needed")
- **Balancing competing constraints**: reliability (prevent outages), cost (minimize resource spend), and performance (maintain SLAs)
- **Making decisions under incomplete information** (metrics are noisy, dependencies are complex)

CloudScalerEnv distills this complexity into a testable benchmark. It challenges agents to:
1. **Anticipate failures** before they happen (memory leaks, CPU saturation)
2. **Coordinate multi-service actions** (respecting service dependencies)
3. **Optimize under constraints** (operate within budget, meet SLA targets)
4. **Recover gracefully** from cascading failures without making things worse

This is **not a game**—it's a realistic simulator used by SRE teams for training and validation. The metrics (CPU, memory, replicas) are literal translations from production Kubernetes dashboards. The grading criteria (uptime %, SLA compliance, cost efficiency) align with engineering KPIs teams actually optimize for.

## Motivation

Production SRE teams continuously balance reliability and cost. They decide whether to scale up to absorb load, scale down to reduce spend, or restart unhealthy services before failures cascade. This environment converts that workflow into a typed, testable benchmark for agent evaluation.

## OpenEnv Interface

The environment implements the required interface:
- reset(...) -> Observation
- step(Action) -> tuple[Observation, Reward, bool, dict]
- state() -> Observation

Typed models:
- Observation: [src/models.py](src/models.py)
- Action: [src/models.py](src/models.py)
- Reward: [src/models.py](src/models.py)

Metadata file:
- [openenv.yaml](openenv.yaml)

## Client and Server

CloudScalerEnv supports both direct server usage and lightweight client SDK usage.

- Server API: [src/app.py](src/app.py)
- Sync/async client wrapper: [src/client.py](src/client.py)

Minimal sync client flow:

```python
from src.client import CloudScalerClient

client = CloudScalerClient(base_url="http://127.0.0.1:7860")
obs = client.reset(task_id="easy-memory-leak", seed=11)
result = client.step({"action_type": "do_nothing"})
state = client.state()
client.close()
```

## Action Space

Action model fields:
- action_type: one of scale_up, scale_down, restart, do_nothing
- service_name: target service name (optional for do_nothing)
- count: replica delta for scaling (1..3)

## Observation Space

Observation includes comprehensive system state:
- **step_count, max_steps**: Episode progress
- **services**: Map of microservices with real-time metrics
  - replicas: Number of active service instances (1-10)
  - cpu_utilization: Average CPU load (0-100%)
  - memory_utilization: Average memory usage including leaks (0-100%)
  - status: health state (healthy, degraded, crashed)
- **total_budget_used**: Cumulative replica-steps (cost metric)
- **total_cost**: Operational overhead from scaling/restart actions
- **crash_count**: Services that have crashed so far
- **restart_count**: Restarts performed by agent
- **invalid_action_count**: Malformed actions (e.g., scaling non-existent service)
- **avg_cpu_deviation**: Deviation from optimal 60% CPU target
- **sla_violations**: Timesteps where CPU was outside 50-70% SLA band
- **uptime_percent**: Overall service availability (0-100%)

## Reward Design

CloudScalerEnv provides a **dense, multi-dimensional reward signal** that guides agents toward realistic SRE objectives:

**Positive Components** (weighted):
- **CPU Balance (25%)**: Reward for keeping services near 60% CPU utilization (optimal zone)
- **Memory Safety (20%)**: Penalty for high memory usage, reward for maintaining margin before crash
- **Reliability (20%)**: Bonus for healthy services, penalty for crashes and degradation
- **Budget Efficiency (15%)**: Reward for fewer replicas (cost control), penalize over-provisioning
- **SLA Compliance (15%)**: Bonus for maintaining CPU in 50-70% service-level band  
- **Cost Efficiency (5%)**: Penalty for expensive actions (scale_up, restart)
- **Uptime Reward**: Cumulative reward for service availability

**Negative Components**:
- **Crash Penalty**: -1.0 per crash (drastic failure signal)
- **Invalid Action Penalty**: -0.5 for malformed actions (learning safety)

The reward is **shaped over the full trajectory**, not just at episode end. This dense signal gives agents immediate feedback on the quality of their decisions, accelerating learning and enabling frontier LLMs to optimize effectively.

## Tasks And Deterministic Graders

All tasks are defined in [src/tasks.py](src/tasks.py) with **multi-dimensional deterministic graders** returning 0.0-1.0 based on six evaluation criteria.

### Task 1: Easy - Memory Leak Prevention (easy-memory-leak)

**Scenario**: A single web frontend service has a memory leak. Memory increases 2-5% per timestep. If it exceeds 95%, the service crashes.

**Objective**: Restart the service before memory saturation causes an outage, while avoiding unnecessary scaling.

**Grader Components** (weighted):
- Reliability (40%): Did you prevent the crash? (1.0 if no crash, 0.0 otherwise)
- Uptime (25%): What % of steps was the service healthy?
- Budget (15%): Did you stay close to target replica budget (~45 replica-steps)?
- Action Cost (10%): Did you minimize restart operations?
- Stability (10%): Was CPU change predictable and smooth?

**Why Hard for Agents**: Requires predictive action (restart before failure), not reactive (after detecting crash).
Max score: **~0.92** with perfect crash prevention and minimal unnecessary restarts.

---

### Task 2: Medium - Traffic Spike Handling (medium-traffic-spike)

**Scenario**: Two dependent services (auth-api → payment-api) face a CPU traffic spike. Auth-api starts at 88% CPU, payment-api at 60%. CPU fluctuates ±15% per step. SLA requires 50-70% CPU band.

**Objective**: Scale services dynamically to maintain SLA compliance while controlling costs. If auth-api crashes, payment-api becomes degraded.

**Grader Components** (weighted):
- Reliability (35%): Prevent crashes 
- SLA Compliance (35%): Maintain 50-70% CPU band for both services
- Budget (15%): Keep replica usage reasonable (~90 replica-steps)
- Action Cost (10%): Minimize scale operations
- Stability (5%): Low CPU deviation

**Service Dependencies**: If auth-api crashes → payment-api degrades (loses requests). Highlights need for coordinated scaling.

**Why Hard for Agents**: Multi-service coordination, SLA-aware decision-making, cost-benefit tradeoffs.
Current calibrated baseline for this task is listed in the Baseline Scores section.

---

### Task 3: Hard - Cascading Failure Mitigation (hard-cascading-failure)

**Scenario**: Three interdependent services (frontend → backend → db-proxy) with tight 30-step budget. All start in stressed states. Cascading failures occur if dependencies crash.

**Objective**: Prevent cascading failures while staying within strict budget limit. One service crash can trigger others.

**Grader Components** (weighted):
- Reliability (30%): Prevent cascade (crash of one service shouldn't kill others)
- Uptime (20%): Maximize availability of all three services
- SLA Compliance (20%): Maintain 50-70% CPU for healthy services
- Budget (15%): Strict cost discipline (~180 replica-steps across 30 steps)
- Action Cost (10%): Minimize operational overhead
- Stability (5%): Predictable, controlled behavior

**Service Dependencies**: 
- frontend depends on backend
- backend depends on db-proxy
- If db-proxy crashes → backend degrades → frontend degrades

**Why Hard for Agents**: Requires understanding of dependency graph, preventive action, multi-step planning under constraints.
Current calibrated baseline for this task is listed in the Baseline Scores section.


## Setup

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. OpenEnv validation

Option A: External CLI (if `openenv` package is installed)

```bash
openenv validate
```

Option B: Local validator (always available)

```bash
python validate_local.py
```

The local validator checks metadata, models, interface contract, and task/grader functionality.

3. Run baseline (heuristic)

```bash
python -m src.baseline --mode heuristic
```

Run baseline with an explicit seed override:

```bash
python -m src.baseline --mode heuristic --seed 999
```

Run a seed sweep to verify scores change across seeds:

```bash
python -m src.baseline --mode heuristic --seed-sweep 11,22,999
```

4. Run baseline (OpenAI API)

```bash
set OPENAI_API_KEY=your_key_here
python -m src.baseline --mode openai --model gpt-4.1-mini
```

## Reset Scenario Configuration

The reset endpoint accepts runtime overrides so task/scenario setup does not require code edits.

Request payload fields:
- `task_id`: `easy-memory-leak`, `medium-traffic-spike`, or `hard-cascading-failure`
- `seed`: override task seed
- `max_steps`: override horizon (must be 5 to 100)
- `services`: optional full service map override

Example:

```json
{
  "task_id": "medium-traffic-spike",
  "seed": 99,
  "max_steps": 30,
  "services": {
    "auth-api": {
      "replicas": 2,
      "cpu_utilization": 75.0,
      "memory_utilization": 45.0,
      "status": "healthy"
    }
  }
}
```

Behavior:
- Missing services stay at task defaults
- Provided services replace/add entries by service name
- Task dependency logic remains task-specific

### Configuration Matrix

| Field | Type | Default | Valid Range/Values | Impact |
|---|---|---|---|---|
| `task_id` | string | `easy-memory-leak` | `easy-memory-leak`, `medium-traffic-spike`, `hard-cascading-failure` | Selects scenario topology, initial state, dependencies, and grader |
| `seed` | integer | task seed (`11`, `22`, `33`) | any integer | Controls stochastic trajectory realization |
| `max_steps` | integer | task default (`20`, `24`, `30`) | `5..100` | Changes episode horizon and budget pressure |
| `services` | object | task initial services | valid `ServiceState` schema per service | Overrides initial replicas/CPU/memory/status per service |
| `services.<name>.replicas` | integer | task default | `1..10` | Capacity and budget usage |
| `services.<name>.cpu_utilization` | float | task default | `0..100` | Immediate SLA pressure and scaling decisions |
| `services.<name>.memory_utilization` | float | task default | `0..100` | Crash risk and restart urgency |
| `services.<name>.status` | string | `healthy` | `healthy`, `degraded`, `crashed` | Starts service in healthy/degraded/crashed mode |

## Submission Contract

This repository is configured to satisfy the evaluator contract.

- Root inference script: `inference.py`
- LLM SDK: `OpenAI` client (`from openai import OpenAI`)
- Required environment variables:
  - `API_BASE_URL` with default value
  - `MODEL_NAME` with default value
  - `HF_TOKEN` required (no default)
- Output line format emitted by `inference.py`:
  - `[START] task=<task_name> env=<benchmark> model=<model_name>`
  - `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
  - `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`
- Numeric formatting:
  - `reward` is formatted to two decimals
  - `rewards` list values are formatted to two decimals

Quick local contract check:

```bash
python pre_submit_check.py
```

This check validates:
- Inference output format compliance
- Environment variable contract
- API smoke test for `/health`, `/reset`, `/step`, and `/state`

## Guideline Compliance Matrix

| Guideline Item | Requirement | Implementation Evidence |
|---|---|---|
| Project structure | `inference.py` at repo root | [inference.py](inference.py) |
| LLM usage | Use OpenAI client for LLM calls | [inference.py](inference.py) imports and uses `OpenAI` |
| Env vars | `API_BASE_URL` default, `MODEL_NAME` default, `HF_TOKEN` required | [inference.py](inference.py) env var definitions and validation |
| Output format | Emit `[START]`, `[STEP]`, `[END]` with exact fields and formatting | [inference.py](inference.py) logging helpers and episode flow |
| End-line guarantee | `[END]` emitted even on exceptions | [inference.py](inference.py) `finally` block |
| API viability | Environment responds on required endpoints | [src/app.py](src/app.py), [pre_submit_check.py](pre_submit_check.py) |
| OpenEnv contract | Typed models and reset/step/state lifecycle | [src/models.py](src/models.py), [src/env.py](src/env.py), [validate_local.py](validate_local.py) |
| Runtime sanity | Lightweight runtime suitable for constrained hardware | [requirements.txt](requirements.txt), [Dockerfile](Dockerfile) |
| Local verification | One-command pre-submit checks | [pre_submit_check.py](pre_submit_check.py) |

## Evaluation Protocol

CloudScalerEnv uses dense per-step reward and deterministic final grading.

What is optimized:
- reliability (crash avoidance)
- SLA compliance (50-70% CPU operating band)
- uptime
- budget and action-cost efficiency
- operational stability

Anti-exploit properties:
- invalid actions are penalized
- excessive scaling/restarts increase cost and reduce score components
- no single metric can dominate without tradeoffs in other metrics

Determinism guarantees:
- grader is deterministic for the realized final state
- trajectory dynamics are stochastic through task seeds and environment noise
- seed sweep support demonstrates non-constant scoring behavior

## Benchmark Score Policy

The benchmark uses a two-layer score policy so training and evaluation are both meaningful:

| Layer | Used For | Signal Type | Determinism | Notes |
|---|---|---|---|---|
| Per-step reward (`Reward.value`) | Learning/optimization loop | Dense process signal | Deterministic from current state/action transition | Encourages safe and efficient decisions at each step |
| Final task grader (`TaskGrader.grade`) | Benchmark comparison/ranking | Terminal evaluation signal | Deterministic from final state | Multi-factor score across reliability, SLA, budget, cost, and stability |

Rubric flow:

```text
trajectory actions -> per-step dense rewards (train signal)
final env state    -> deterministic multi-factor grade (eval score)
```

## Execution Modes

CloudScalerEnv supports two practical execution modes:

- API server mode
  - run FastAPI service and interact over `/reset`, `/step`, `/state`
  - best for deployment and evaluator parity
- Local module mode
  - run local scripts against `src.env.CloudScalerEnv`
  - best for development and rapid iteration

### Resource Behavior Under 2 vCPU and 8 GB RAM

The project is intentionally lightweight:
- CPU-only runtime
- small dependency footprint
- short inference prompt and bounded token usage

Operational guidance:
- keep only one active Hugging Face Space during submission
- avoid heavy background processes in container startup
- use `python pre_submit_check.py` before each submission to catch regressions

## Examples

Runnable examples are included in [examples](examples):

- [examples/manual_episode.py](examples/manual_episode.py)
  - step-by-step manual control episode through API client
- [examples/heuristic_episode.py](examples/heuristic_episode.py)
  - end-to-end heuristic run with printed trajectory summary
- [examples/inference_contract_check.py](examples/inference_contract_check.py)
  - runs pre-submit inference/API contract checks
- [examples/async_client_episode.py](examples/async_client_episode.py)
  - async client usage with automatic local server lifecycle

Run examples:

```bash
python examples/manual_episode.py
python examples/heuristic_episode.py
python examples/inference_contract_check.py
python examples/async_client_episode.py
```

Competition dry-run summary in one command:

```bash
python competition_dry_run.py
```

## Testing

Install test dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

Included tests cover:
- reset overrides behavior
- inference output contract shape
- deterministic grading for identical seeds and trajectories
- API edge cases (invalid action type, missing fields, unknown task, step/state before reset)

Run concurrent stress testing:

```bash
python stress_test.py --task medium-traffic-spike --concurrency 20 --requests-per-worker 20 --output stress_report.json
```

This command validates concurrent `/reset`, `/step`, and `/state` request groups and writes a machine-readable report to `stress_report.json`.

## Submission Runbook

1. Stop all non-primary Hugging Face Spaces.
2. Ensure primary Space build is complete and status is `Running`.
3. Run local checks:
  - `python pre_submit_check.py`
  - `python validate_local.py`
  - `python competition_dry_run.py`
4. Push final code and re-check Space `/health` and `/reset`.
5. Submit only after live endpoint check succeeds.
6. If validation fails, fix and resubmit (no penalty).

## Baseline Scores

Deterministic heuristic baseline (current implementation):
- Easy: 0.9588
- Medium: 0.8025
- Hard: 0.8699
- Mean: 0.8770

OpenAI baseline is reproducible with fixed task seeds and temperature 0.

## Performance Guarantees

The service is designed to stay lightweight and reliable under the competition constraints.

Recent local stress test (`python stress_test.py --task medium-traffic-spike --concurrency 20 --requests-per-worker 20`):

- request groups: 400
- success rate: 100.0%
- duration: 4.282s
- throughput: 93.42 request-groups/sec
- latency p50: 161.79ms
- latency p95: 517.08ms
- latency p99: 694.98ms

Request-group definition:
- one `POST /reset`
- one `POST /step`
- one `GET /state`

Operational guarantees:
- deterministic grading for identical realized trajectories
- graceful handling of malformed action payloads via schema validation (422)
- explicit client-side usage errors for `step/state` before `reset` (400)
- inference path fails fast on unreachable model endpoints (short timeout + no retries)

## Competition FAQ

Q: Why do scores change across tasks and seeds?
A: Environment transitions are seeded stochastic processes. Different seeds produce different trajectories. For a fixed seed and identical action trajectory, the grader is deterministic.

Q: Why are scores not always near 1.0?
A: The benchmark intentionally enforces trade-offs among reliability, SLA compliance, and cost. Perfect uptime with overspending or unstable actions is penalized.

Q: How is this protected from reward hacking?
A: The reward and final grader both penalize invalid actions, excessive action cost, and poor SLA adherence. No single metric can dominate score quality.

Q: What if a model endpoint is unavailable at runtime?
A: Inference uses bounded timeout and zero retries for fast failure and then safely falls back to deterministic local policy behavior.

Q: How can reviewers quickly verify submission health?
A: Run:
`python pre_submit_check.py`
`python validate_local.py`
`python -m pytest -q`
`python competition_dry_run.py`

Q: How do I verify concurrent reliability?
A: Run `python stress_test.py --task medium-traffic-spike --concurrency 20 --requests-per-worker 20` and inspect `stress_report.json`.

## Validation Results

The project includes a local OpenEnv compliance validator (`validate_local.py`) that checks:
- Metadata structure and required fields in openenv.yaml
- Pydantic model instantiation and validity
- Environment interface (reset, step, state methods)
- Episode lifecycle contract (types and shapes)
- Task loading and deterministic grader output (0.0-1.0 range)

Local validation output (all 5 checks pass):
```
============================================================
CloudScalerEnv Local OpenEnv Validator
============================================================
Validating metadata (openenv.yaml)...
  ✓ Metadata valid
Validating models...
  ✓ ServiceState instantiated
  ✓ Observation instantiated
  ✓ Action instantiated
  ✓ Reward instantiated
Validating environment interface...
  ✓ All required methods present
Validating episode contract...
  ✓ reset() returned Observation
  ✓ step() contract valid: obs, reward, done, info
  ✓ state() returned Observation
Validating tasks and graders...
  ✓ easy: task loaded, grader score 0.980
  ✓ medium: task loaded, grader score 0.829
  ✓ hard: task loaded, grader score 0.867
============================================================
Result: 5/5 checks passed
============================================================
```

## Docker

Build and run locally:

```bash
docker build -t cloudscalerenv .
docker run --rm -p 7860:7860 cloudscalerenv
```

Then verify the service:

```bash
curl http://localhost:7860/health
curl "http://localhost:7860/baseline?mode=heuristic"
```
