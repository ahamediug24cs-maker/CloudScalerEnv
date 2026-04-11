# 🚀 Submission Checklist

## Pre-Submission Validation

### 1. Environment Setup (30 seconds)
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set required env vars
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-huggingface-token-here"
export TASK_NAME="easy-memory-leak"  # or medium-traffic-spike, hard-cascading-failure
```

### 2. Local Validation (2 minutes)
```bash
# Run all validation checks
python pre_submit_check.py         # ✓ Contract + environment + API smoke test
python validate_local.py            # ✓ Schema + graders + determinism
python -m pytest -q                 # ✓ Unit tests (5 tests)
python competition_dry_run.py       # ✓ Aggregate verification (PASS/FAIL)
```

### 3. Example Execution (1 minute)
```bash
# Quick exec test
python examples/manual_episode.py   # ✓ End-to-end flow
python examples/heuristic_episode.py # ✓ Alternative policy
python examples/async_client_episode.py # ✓ Async support
```

### 4. Performance Profile (< 30 seconds)
```bash
# Verify resource constraints
python -m src.baseline --mode heuristic  # Easy 0.96+, Medium 0.80+, Hard 0.87+, Mean 0.88+
```

## Submission Artifacts

### Required Files
- ✅ `inference.py` — Root-level inference contract with [START]/[STEP]/[END]
- ✅ `openenv.yaml` — Environment metadata + task specs
- ✅ `requirements.txt` — Dependencies (openai, pydantic, fastapi, etc.)
- ✅ `src/` — Core source directory

### Optional (Recommended for Competitive Edge)
- ✅ `README.md` — Architecture, usage, compliance matrix
- ✅ `SUBMISSION_DEFENSE_GUIDE.md` — Rubric alignment, design decisions
- ✅ `pre_submit_check.py` — Automation for reviewers
- ✅ `baseline_comparison.py` — Evidence of 17.7% competitive advantage
- ✅ `tests/` — Pytest suite for reproducibility

## Environment Variables Required

| Var | Required | Default | Notes |
|-----|----------|---------|-------|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | Yes | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | (none) | Hugging Face API key (must provide) |
| `TASK_NAME` | No | `easy-memory-leak` | Task override (easy/medium/hard) |
| `BENCHMARK_NAME` | No | `cloudscalerenv` | Benchmark identifier |

## Compliance Matrix

| Requirement | Status | Evidence |
|------------|--------|----------|
| Inference contract ([START]/[STEP]/[END]) | ✅ | pre_submit_check.py passes |
| OpenAI client required | ✅ | inference.py:7 |
| Env vars (API_BASE_URL, MODEL_NAME, HF_TOKEN) | ✅ | inference.py:14-15, 16 |
| 2vCPU/8GB resource compliance | ✅ | Lightweight, Uvicorn startup < 2s |
| 60-second inference timeout | ✅ | OpenAI client: timeout=5.0, max_retries=0 |
| Deterministic graders (same seed = same score) | ✅ | test_grader_determinism.py |
| Handle missing env vars gracefully | ✅ | Raises "HF_TOKEN required" message |

## Execution Commands

### Standard Inference (What Evaluator Runs)
```bash
python inference.py
```
Expected output: Deterministic [START], [STEP] lines, [END] with rewards.

### API Server (Standalone Testing)
```bash
python -m src.app  # Starts Uvicorn on http://127.0.0.1:7860
```

### Full QA Suite
```bash
python competition_dry_run.py  # Runs all checks + baseline + validator
```

## Scoring Expectations

Baseline heuristic policy performance (seed fixed):
- **Easy**: 0.96+ (memory leak proactive restart)
- **Medium**: 0.80+ (multi-service SLA compliance)
- **Hard**: 0.87+ (cascading failure prevention)
- **Mean**: 0.88+ across all tasks

Competitive advantage:
- **17.7%** better than random agent
- **24%** better than do-nothing baseline

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| UnicodeEncodeError (Windows) | Set `PYTHONUTF8=1` before running |
| API timeout | Check `HF_TOKEN` valid and network connectivity |
| Import errors | Run `pip install -r requirements.txt` |
| Tests fail | Run `python -m pytest tests/ -v` for debug output |
| Scores lower than expected | Seed may vary; run validate_local.py 3x to verify |

## Contact & Support

For issues, check:
1. `README.md` — Architecture and usage guide
2. `SUBMISSION_DEFENSE_GUIDE.md` — Design rationale
3. `tests/` — Unit tests for reference behavior
4. Run `python pre_submit_check.py` — Comprehensive validation

---

**Ready to Submit!** ✅ All tests passing, competitive baselines validated.
