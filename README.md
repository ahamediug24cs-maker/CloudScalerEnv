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

CloudScalerEnv is a production-grade SRE simulation environment where AI agents learn to autonomously manage microservice infrastructure under realistic stress conditions. The environment models the critical decision-making workflows that SRE teams perform daily: scaling services to handle traffic spikes, restarting unhealthy services, managing cascading failures, and maintaining tight cost budgets—all while preventing service outages.

## Motivation: Why This Matters

Modern cloud-native systems require 24/7 SRE oversight. Production SRE teams spend significant time:
- **Detecting abnormal metrics** (CPU spikes, memory leaks) and responding within minutes
- **Orchestrating multi-step remediation** (e.g., "scale this service, monitor for cascade effects, then restart if needed")
- **Balancing competing constraints**: reliability (prevent outages), cost (minimize resource spend), and performance (maintain SLAs)
- **Making decisions under incomplete information** (metrics are noisy, dependencies are complex)

CloudScalerEnv distills this complexity into a testable benchmark. It challenges AI agents to:
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
Observed heuristic baseline: **~0.52** with the current policy and grader calibration.

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
Observed heuristic baseline: **~0.58** with the current policy and grader calibration.


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

## Baseline Scores

Deterministic heuristic baseline (current implementation):
- Easy: 0.725
- Medium: 0.516
- Hard: 0.584
- Mean: 0.608

OpenAI baseline is reproducible with fixed task seeds and temperature 0.

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
  ✓ easy: task loaded, grader score 0.269
  ✓ medium: task loaded, grader score 0.325
  ✓ hard: task loaded, grader score 0.261
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
