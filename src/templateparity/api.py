from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .experiments import synthetic_demo
from .fixtures.loader import list_fixtures, load_fixture
from .reporting import markdown_report
from .runtimes.llama_adapter import validate_local_base_url
from .storage import ArtifactStore

app = FastAPI(title="LLM Chat Template Compatibility", version="0.1.0")
store = ArtifactStore(".templateparity/runs")


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixture: str
    mode: Literal["native", "shared", "crossover"] = "native"
    mlx_model: str = Field(min_length=1)
    llama_url: str = "http://127.0.0.1:8080"
    max_tokens: int = Field(default=128, ge=1, le=4096)
    repeat: int = Field(default=1, ge=1, le=10)
    no_persist_raw: bool = False

    @field_validator("fixture")
    @classmethod
    def known_fixture(cls, value):
        if value not in list_fixtures():
            raise ValueError("Unknown fixture identifier")
        return value

    @field_validator("llama_url")
    @classmethod
    def loopback_only(cls, value):
        return validate_local_base_url(value)


@app.get("/api/health")
def health():
    return {"status": "ok", "local_only": True}


@app.get("/api/runtimes")
def runtimes():
    return [
        {"name": "MLX-LM", "availability": "Apple Silicon optional dependency"},
        {"name": "llama.cpp", "availability": "local llama-server"},
    ]


@app.get("/api/models")
def models():
    return {"registered": [], "note": "No models are downloaded or registered automatically."}


@app.get("/api/fixtures")
def fixtures():
    return [
        {"id": name, "conversation": load_fixture(name).model_dump(mode="json")}
        for name in list_fixtures()
    ]


@app.get("/api/experiments")
def experiments():
    return [{"id": name} for name in store.list()]


@app.get("/api/demo")
def demo():
    return synthetic_demo().model_dump(mode="json")


@app.post("/api/experiments")
def create_experiment(request: CompareRequest):
    from .cli import compare

    previous = set(store.list())
    try:
        compare(
            fixture=request.fixture,
            mode=request.mode,
            mlx_model=Path(request.mlx_model),
            llama_url=request.llama_url,
            max_tokens=request.max_tokens,
            repeat=request.repeat,
            no_persist_raw=request.no_persist_raw,
            output_dir=Path(".templateparity/runs"),
        )
    except typer.Exit as exc:
        created = sorted(set(store.list()) - previous)
        if created:
            identifier = created[-1]
            return {"id": identifier, "status": "blocked", "exit_code": exc.exit_code}
        raise HTTPException(
            503, "Experiment could not start; inspect local server output."
        ) from exc
    created = sorted(set(store.list()) - previous)
    if not created:
        raise HTTPException(500, "Experiment finished without writing an artifact.")
    return {"id": created[-1], "status": "completed"}


def _load(experiment_id: str):
    try:
        return store.load(experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Experiment not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id: str):
    return _load(experiment_id).model_dump(mode="json")


@app.get("/api/experiments/{experiment_id}/status")
def experiment_status(experiment_id: str):
    _load(experiment_id)
    return {"id": experiment_id, "status": "completed"}


@app.get("/api/experiments/{experiment_id}/report")
def report(experiment_id: str):
    return {"markdown": markdown_report(_load(experiment_id))}


@app.get("/api/experiments/{experiment_id}/artifacts")
def experiment_artifacts(experiment_id: str):
    experiment = _load(experiment_id)
    return {
        "id": experiment_id,
        "json": experiment.model_dump(mode="json"),
        "markdown": markdown_report(experiment),
    }


_frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _frontend.is_dir():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
