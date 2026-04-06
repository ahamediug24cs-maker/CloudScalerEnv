from fastapi import FastAPI, HTTPException, Query

from .baseline import run_baseline


app = FastAPI(
    title="CloudScalerEnv Service",
    description="HF Space runtime for CloudScalerEnv with health and baseline endpoints.",
    version="1.0.0",
)


@app.get("/")
def root() -> dict:
    return {
        "name": "CloudScalerEnv",
        "status": "ok",
        "message": "Service is running.",
        "endpoints": {
            "health": "/health",
            "baseline": "/baseline?mode=heuristic",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"healthy": True}


@app.get("/baseline")
def baseline(mode: str = Query("heuristic", pattern="^(heuristic|openai)$"), model: str = "gpt-4.1-mini") -> dict:
    if mode == "openai":
        raise HTTPException(
            status_code=400,
            detail="OpenAI mode should be executed from CLI with OPENAI_API_KEY to avoid exposing credentials.",
        )

    scores = run_baseline(mode=mode, model=model)
    mean_score = round(sum(scores.values()) / len(scores), 3)
    return {"mode": mode, "scores": scores, "mean_score": mean_score}
