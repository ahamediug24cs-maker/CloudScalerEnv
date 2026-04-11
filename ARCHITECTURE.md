# Architecture & Design Flow

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenEnv Competition Entry                    │
│                                                                   │
│  Task Spec (Task ID, Difficulty, Objective, Max Steps)          │
│           ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │         CloudScalerEnv (OpenEnv Interface)           │       │
│  │                                                        │       │
│  │  reset(task) → Observation  [Initializes state]      │       │
│  │  step(action) → (Obs, Reward, Done, Info)            │       │
│  │  state() → Observation      [Query current state]     │       │
│  │                                                        │       │
│  │  Environment Dynamics:                                │       │
│  │  • Memory leaks (+2-5% per step, CPU-amplified)      │       │
│  │  • Traffic spikes (-5% to +15% CPU variance)          │       │
│  │  • Service dependencies (cascading failures)          │       │
│  │  • Budget constraints (replicas per max_steps)        │       │
│  │  • SLA violations (CPU target 50-70%)                │       │
│  │                                                        │       │
│  └──────────────────────────────────────────────────────┘       │
│           ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │      Observation (Strongly Typed State)               │       │
│  │                                                        │       │
│  │  • Services[] {replicas, cpu, memory, status}         │       │
│  │  • step_count, total_budget_used, crash_count         │       │
│  │  • sla_violations, uptime_percent, avg_cpu_deviation │       │
│  │                                                        │       │
│  └──────────────────────────────────────────────────────┘       │
│           ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │       Agent Decision Logic (Policy / Model)           │       │
│  │                                                        │       │
│  │  Heuristic Policy (included baseline):                │       │
│  │  • CPU threshold prediction for proactive scaling     │       │
│  │  • SLA-aware thresholds (adapt based on violations)   │       │
│  │  • Crash history→cascade prevention (lower thresholds)│       │
│  │  • Memory-triggered restarts                          │       │
│  │                                                        │       │
│  │  LLM Policy (competition inference.py):                │       │
│  │  • Receive observation as JSON prompt                │       │
│  │  • Generate action (scale_up/scale_down/restart) via OpenAI  │       │
│  │  • Fall back to heuristic on parse failures           │       │
│  │                                                        │       │
│  └──────────────────────────────────────────────────────┘       │
│           ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │       Action (Strongly Typed Command)                 │       │
│  │                                                        │       │
│  │  • action_type: scale_up | scale_down | restart       │       │
│  │  • service_name: which service to target              │       │
│  │  • count: how many replicas to add/remove (1-3)      │       │
│  │                                                        │       │
│  └──────────────────────────────────────────────────────┘       │
│           ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │       Reward Function (Dense Signal)                  │       │
│  │                                                        │       │
│  │  State Metrics (70% weight):                          │       │
│  │  • cpu_balance (stay near 60% target)                │       │
│  │  • memory_safety (keep utilization low)              │       │
│  │  • reliability (prefer healthy status)               │       │
│  │  • budget_efficiency (fewer replicas better)         │       │
│  │  • sla_compliance (CPU 50-70% range)                │       │
│  │  • cost_efficiency (low action costs)                │       │
│  │  • uptime_percent (availability)                     │       │
│  │                                                        │       │
│  │  Decision Quality Metrics (30% weight):              │       │
│  │  • timeliness (proactive vs reactive)               │       │
│  │  • action_efficiency (minimal actions)              │       │
│  │                                                        │       │
│  │  Invalid_action_penalty: -0.5 per malformed command  │       │
│  │                                                        │       │
│  └──────────────────────────────────────────────────────┘       │
│           ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │       TaskGrader (Final Episode Score)                │       │
│  │                                                        │       │
│  │  Difficulty-Specific Scoring:                         │       │
│  │                                                        │       │
│  │  Easy (Memory Leak):                                  │       │
│  │  • Weights: reliability (36%), timeliness (18%),      │       │
│  │    uptime (18%), budget (10%), others (18%)           │       │
│  │  • Calibration: 0.30 + 0.70 * raw_score             │       │
│  │  • Target: Restart before crash                      │       │
│  │                                                        │       │
│  │  Medium (Traffic Spike):                              │       │
│  │  • Weights: reliability (28%), SLA (24%),            │       │
│  │    timeliness (14%), budget (16%), others (18%)       │       │
│  │  • Calibration: 0.50 + 0.50 * raw_score             │       │
│  │  • Target: Keep CPU 50-70%, scale efficiently        │       │
│  │                                                        │       │
│  │  Hard (Cascading Failures):                           │       │
│  │  • Weights: reliability (26%), uptime (20%),         │       │
│  │    timeliness (14%), others (40%)                     │       │
│  │  • Calibration: 0.48 + 0.52 * raw_score             │       │
│  │  • Target: Prevent cascades across services          │       │
│  │                                                        │       │
│  │  Properties:                                          │       │
│  │  ✓ Deterministic: same trajectory → same score       │       │
│  │  ✓ Sensitive: trajectory quality matters (action counts)      │       │
│  │  ✓ Calibrated: difficulty-normalized 0.0-1.0 range  │       │
│  │                                                        │       │
│  └──────────────────────────────────────────────────────┘       │
│           ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │       Episode Score (0.0 to 1.0)                      │       │
│  │                                                        │       │
│  │  Competitive Baseline:                                │       │
│  │  • Easy: 0.96+   (heuristic consistently high)       │       │
│  │  • Medium: 0.80+ (SLA-aware scaling works)           │       │
│  │  • Hard: 0.87+   (proactive cascade prevention)      │       │
│  │  • Mean: 0.88+   across difficult scenarios          │       │
│  │                                                        │       │
│  │  Advantage vs Baselines:                              │       │
│  │  • 17.7% better than random agent                    │       │
│  │  • 24% better than do-nothing agent                  │       │
│  │                                                        │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## API Server Flow (FastAPI)

```
Client Request
    ↓
┌─────────────────────────┐
│  POST /reset {task_id}  │  (optional: max_steps, services, seed)
│      ↓                  │
│  Initialize env         │
│  reset_for_task()       │
│      ↓                  │
│  Return Observation     │ ← Reset step_count=0
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  POST /step {action}    │  (action_type, service_name, count)
│      ↓                  │
│  Validate action schema │
│  env.step(action)       │
│      ↓                  │
│  Return (Obs, Reward,   │ ← Dense signal with decision metrics
│           Done, Info)   │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  GET /state             │
│      ↓                  │
│  Return current Obs     │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  GET /health            │
│      ↓                  │
│  Return {healthy: bool} │
└─────────────────────────┘
```

## Inference Contract (What LLM Sees)

```
ENVIRONMENT CONTEXT:
{
  "step": 1,
  "max_steps": 20,
  "services": [
    {
      "name": "web-frontend",
      "replicas": 2,
      "cpu": 64.50,
      "memory": 85.00,
      "status": "healthy"
    }
  ],
  "total_budget_used": 180.50,
  "crash_count": 0,
  "restart_count": 1,
  "invalid_action_count": 0
}

LLM MUST OUTPUT (JSON):
{
  "action_type": "scale_up",        # or scale_down, restart, do_nothing
  "service_name": "web-frontend",
  "count": 1
}

SYSTEM PROMPT (Concise):
"You are an SRE agent. Choose one action to improve reliability and cost efficiency.
 Return ONLY one compact JSON with keys: action_type, service_name, count.
 action_type must be one of: scale_up, scale_down, restart, do_nothing."

STDOUT PROTOCOL (Required):
[START] task=easy-memory-leak env=cloudscalerenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=do_nothing reward=0.47 done=false error=null
[STEP] step=2 action=restart(web-frontend,1) reward=0.92 done=false error=null
...
[END] success=true steps=20 rewards=0.47,0.92,0.93,...,0.76
```

## Data Flow Summary

```
Determinism & Reproducibility:
  Task + Seed → Env State (reproducible)
            ↓
  Policy + Obs → Action (policy-specific)
            ↓
  Action → Dynamics (random but seeded)
            ↓
  Final State → Score (deterministic grader)

Sensitivity:
  Same Final State + Different Trajectory
                ↓
  Action sequence matters (timeliness/efficiency weights)
                ↓
  Scores differ based on decision quality
```

---

This architecture ensures:
- ✅ **Reproducibility**: Seeded RNG, deterministic grading
- ✅ **Fairness**: Consistent scoring across runs for same trajectory
- ✅ **Responsiveness**: Granular reward + decision quality metrics
- ✅ **Realism**: Memory-CPU interactions, cascading failures, budget constraints
- ✅ **Competitiveness**: 17.7% advantage over baselines
