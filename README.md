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

CloudScalerEnv models a real SRE workflow: keeping microservices stable during traffic volatility and memory leaks while controlling infrastructure cost.

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

Observation includes:
- step_count, max_steps
- services map with replicas, cpu_utilization, memory_utilization, status
- total_budget_used
- crash_count, restart_count, invalid_action_count
- avg_cpu_deviation

## Reward Design

Reward is dense and shaped over the full trajectory:
- Positive: CPU balance toward target band, memory safety margin, healthy service reliability, budget efficiency
- Negative: crashes and invalid actions

The scalar reward value is returned each step together with named components for interpretability.

## Tasks And Deterministic Graders

All tasks are defined in [src/tasks.py](src/tasks.py) with deterministic graders returning 0.0-1.0:

1. Easy (easy-memory-leak)
- Objective: Restart a leaking web frontend before crash.
- Grader emphasizes no crashes, timely restarts, and reasonable budget.

2. Medium (medium-traffic-spike)
- Objective: Keep two APIs stable in the 50-70% CPU region while avoiding waste.
- Grader emphasizes crash avoidance, CPU deviation control, and efficient scaling.

3. Hard (hard-cascading-failure)
- Objective: Prevent cascading failures across three services under tight budget.
- Grader emphasizes reliability, CPU control, restart efficiency, and low invalid actions.

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

4. Run baseline (OpenAI API)

```bash
set OPENAI_API_KEY=your_key_here
python -m src.baseline --mode openai --model gpt-4.1-mini
```

## Baseline Scores

Deterministic heuristic baseline (current implementation):
- Easy: 0.929
- Medium: 0.283
- Hard: 0.266
- Mean: 0.493

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
