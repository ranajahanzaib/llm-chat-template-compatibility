from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import uuid
from pathlib import Path

import typer

from . import __version__
from .comparison import compare_prompts, diagnose
from .experiments import synthetic_demo
from .fixtures.loader import list_fixtures, load_fixture
from .models import Experiment, Finding, RuntimeConfiguration
from .reporting import markdown_report
from .runtimes import LlamaCppAdapter, MLXAdapter
from .storage import ArtifactStore, redact_experiment
from .tokenization import assess_tokenizer_compatibility

app = typer.Typer(
    no_args_is_help=True,
    help="Find prompt and token differences behind inconsistent local LLM behavior.",
)
model_app = typer.Typer(
    no_args_is_help=True, help="Inspect and register local model preparation manifests."
)
app.add_typer(model_app, name="models")


def _version_callback(ctx, param, value):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version_flag: bool = typer.Option(None, "--version", callback=_version_callback, is_eager=True),
):
    """Find prompt and token differences behind inconsistent local LLM behavior."""
    del version_flag


PROBE_TEXTS = {
    "basic": "LLM Chat Template Compatibility tokenizer probe 123.",
    "unicode": "café 🛰️",
    "whitespace": "line one\nline two",
}


def _emit(data, json_output: bool) -> None:
    typer.echo(json.dumps(data, indent=2, default=str) if json_output else data)


def _save(experiment: Experiment, output_dir: Path, persist_raw: bool = True) -> Path:
    persisted = experiment if persist_raw else redact_experiment(experiment)
    path = ArtifactStore(output_dir).save(persisted)
    path.with_suffix(".md").write_text(markdown_report(persisted), encoding="utf-8")
    return path


def _tokenizer_checks(mlx, llama, prompts):
    left = {key: mlx.tokenize(value) for key, value in PROBE_TEXTS.items()}
    right = {key: llama.tokenize(value) for key, value in PROBE_TEXTS.items()}
    for key, prompt in prompts.items():
        left[f"prompt:{key}"] = mlx.tokenize(prompt.rendered_text)
        right[f"prompt:{key}"] = llama.tokenize(prompt.rendered_text)
    return (
        assess_tokenizer_compatibility(
            prompts["mlx"].tokenizer_settings,
            prompts["llama"].tokenizer_settings,
            left,
            right,
        ),
        left,
        right,
    )


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_manifest(model_path: Path | None):
    if model_path is None:
        return None
    candidate = model_path.resolve().parent / "model-manifest.json"
    if not candidate.is_file():
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if data.get("schema_version") == "1.0" else None


def _runtime_configs(mlx_model, llama, prompts, max_tokens):
    mlx_manifest = _read_manifest(mlx_model)
    llama_path = Path(llama.props["model_path"]) if llama.props.get("model_path") else None
    llama_manifest = _read_manifest(llama_path)
    same_manifest = bool(mlx_manifest and llama_manifest and mlx_manifest == llama_manifest)
    common = mlx_manifest if same_manifest else None
    declared_id = common.get("source_model_id") if common else None
    declared_revision = common.get("source_revision") if common else None
    runtime_configs = {
        "mlx": RuntimeConfiguration(
            runtime="MLX-LM",
            model_path=str(mlx_model.resolve()),
            source_model_id=mlx_manifest.get("source_model_id") if mlx_manifest else None,
            source_revision=mlx_manifest.get("source_revision") if mlx_manifest else None,
            conversion_metadata={
                "dtype": mlx_manifest.get("mlx_dtype"),
                "quantization": mlx_manifest.get("quantization"),
            }
            if mlx_manifest
            else {},
            tokenizer_metadata=prompts["mlx"].tokenizer_settings,
            template_source=prompts["mlx"].template_source,
            template_options=prompts["mlx"].template_options,
            generation_options={"max_tokens": max_tokens, "sampling": "greedy"},
            runtime_version=_package_version("mlx-lm"),
        ),
        "llama": RuntimeConfiguration(
            runtime="llama.cpp",
            model_path=str(llama_path) if llama_path else None,
            source_model_id=llama_manifest.get("source_model_id") if llama_manifest else None,
            source_revision=llama_manifest.get("source_revision") if llama_manifest else None,
            conversion_metadata={
                "dtype": llama_manifest.get("gguf_dtype"),
                "quantization": llama_manifest.get("quantization"),
            }
            if llama_manifest
            else {},
            tokenizer_metadata=prompts["llama"].tokenizer_settings,
            template_source=prompts["llama"].template_source,
            template_options=prompts["llama"].template_options,
            generation_options={"max_tokens": max_tokens, "temperature": 0, "top_k": 1, "top_p": 1},
            runtime_version=str(llama.props.get("build_info"))
            if llama.props.get("build_info")
            else None,
        ),
    }
    provenance = {
        "source_model_id": declared_id,
        "source_revision": declared_revision,
        "manifest_match": same_manifest,
        "provenance_status": "same-manifest" if same_manifest else "unverified",
    }
    return runtime_configs, provenance, mlx_manifest, llama_manifest


def _blocked_experiment(mode, prompts, compatibility, output_dir, detail, persist_raw=True):
    finding = Finding(
        code="TP011",
        severity="error" if compatibility.status == "mismatch" else "warning",
        category="compatibility",
        evidence="; ".join((*compatibility.warnings, detail)),
        interpretation="Cross-runtime token IDs are not safe to share without matching metadata and checked-vector evidence.",
        suggested_action="Inspect the tokenizer metadata, or use a server build that exposes the required vocabulary details.",
        confidence="observed",
    )
    experiment = Experiment(
        experiment_id=uuid.uuid4().hex[:12],
        mode=mode,
        prompts=prompts,
        token_comparison=compare_prompts(prompts["mlx"], prompts["llama"], False),
        compatibility=compatibility.status,
        diagnostic_findings=[finding],
        conclusion=f"{mode.title()} generation was skipped: {detail}",
        uncertainty=list(compatibility.warnings),
    )
    path = _save(experiment, output_dir, persist_raw)
    typer.echo(f"{experiment.conclusion} Capture saved to {path}", err=True)
    raise typer.Exit(3)


@app.command()
def version():
    typer.echo(__version__)


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json")):
    url = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
    try:
        probe = LlamaCppAdapter(url).probe()
    except ValueError as exc:
        probe = {"ready": False, "runtime": "llama.cpp", "error": str(exc)}
    mlx_available = (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    )
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "llama_cpp": probe,
        "mlx_lm_available": mlx_available,
        "model_artifacts": "Not registered; no automatic downloads",
    }
    _emit(
        result
        if json_output
        else f"Python: {result['python']}\nMLX-LM available: {mlx_available}\n"
        f"llama.cpp ready: {probe['ready']}\n"
        f"{'' if probe['ready'] else probe['error']}\nModel artifacts: not registered",
        json_output,
    )


@app.command("fixtures")
def fixtures(action: str = typer.Argument("list")):
    if action != "list":
        raise typer.BadParameter("The only supported fixtures action is list")
    for fixture in list_fixtures():
        typer.echo(f"{fixture}: {load_fixture(fixture).description}")


@app.command()
def capture(
    fixture: str = typer.Option(...),
    runtime: str = typer.Option(..., help="Runtime to capture: mlx or llama."),
    mlx_model: Path | None = typer.Option(None),
    llama_url: str = typer.Option("http://127.0.0.1:8080"),
    output_dir: Path = typer.Option(Path(".templateparity/runs")),
    no_persist_raw: bool = typer.Option(False, "--no-persist-raw"),
    json_output: bool = typer.Option(False, "--json"),
):
    conversation = load_fixture(fixture)
    if runtime not in {"mlx", "llama"}:
        raise typer.BadParameter("runtime must be mlx or llama")
    try:
        if runtime == "mlx":
            if mlx_model is None or not mlx_model.is_dir():
                raise typer.BadParameter("MLX capture requires an existing --mlx-model directory")
            adapter = MLXAdapter(mlx_model)
        else:
            adapter = LlamaCppAdapter(llama_url)
            readiness = adapter.probe()
            if not readiness["ready"]:
                raise typer.BadParameter(readiness["error"])
        prompt = adapter.render_chat(conversation)
    except typer.BadParameter:
        raise
    except Exception as exc:
        typer.echo(f"Prompt capture failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    runtime_configuration = RuntimeConfiguration(
        runtime=prompt.source_runtime,
        model_path=str(mlx_model.resolve())
        if runtime == "mlx" and mlx_model
        else adapter.props.get("model_path")
        if runtime == "llama"
        else None,
        tokenizer_metadata=prompt.tokenizer_settings,
        template_source=prompt.template_source,
        template_options=prompt.template_options,
        runtime_version=_package_version("mlx-lm")
        if runtime == "mlx"
        else str(adapter.props.get("build_info")),
    )
    experiment = Experiment(
        experiment_id=uuid.uuid4().hex[:12],
        mode="capture",
        prompts={runtime: prompt},
        runtime_configurations={runtime: runtime_configuration},
        compatibility="single-runtime capture",
        conclusion="Rendered prompt and token IDs were captured without running generation.",
        uncertainty=prompt.warnings,
    )
    path = _save(experiment, output_dir, not no_persist_raw)
    _emit(
        {
            "experiment_id": experiment.experiment_id,
            "artifact": str(path),
            "token_count": prompt.token_count,
            "runtime": prompt.source_runtime,
        }
        if json_output
        else f"Captured {prompt.token_count} observed tokens from {prompt.source_runtime}. Saved {path}",
        json_output,
    )


@app.command()
def demo(
    output_dir: Path = typer.Option(Path(".templateparity/runs")),
    json_output: bool = typer.Option(False, "--json"),
):
    experiment = synthetic_demo()
    path = _save(experiment, output_dir)
    _emit(
        {
            "experiment_id": experiment.experiment_id,
            "json": str(path),
            "markdown": str(path.with_suffix(".md")),
            "label": "Synthetic demonstration — not measured model inference.",
        }
        if json_output
        else f"Synthetic demonstration — not measured model inference.\nSaved {path}\nSaved {path.with_suffix('.md')}",
        json_output,
    )


@app.command()
def report(
    run_id: str,
    output_dir: Path = typer.Option(Path(".templateparity/runs")),
    json_output: bool = typer.Option(False, "--json"),
):
    experiment = ArtifactStore(output_dir).load(run_id)
    _emit(
        experiment.model_dump(mode="json") if json_output else markdown_report(experiment),
        json_output,
    )


@model_app.command("register")
def register_model(manifest: Path = typer.Option(..., "--manifest")):
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Cannot read model manifest: {exc}") from exc
    revision = data.get("source_revision", "")
    if (
        data.get("schema_version") != "1.0"
        or len(revision) != 40
        or any(ch not in "0123456789abcdefABCDEF" for ch in revision)
    ):
        raise typer.BadParameter(
            "Manifest must use schema 1.0 and declare a full source commit SHA."
        )
    for key in ("source_path", "mlx_path", "gguf_path"):
        if not data.get(key) or not Path(data[key]).exists():
            raise typer.BadParameter(f"Manifest artifact path is missing or unavailable: {key}")
    registry = Path(".templateparity/model-manifests")
    registry.mkdir(parents=True, exist_ok=True)
    safe_repo = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in data.get("source_model_id", "model")
    )
    target = registry / f"{safe_repo}-{revision[:12]}.json"
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Registered declared model provenance: {target}")


@model_app.command("inspect")
def inspect_models(manifest: Path | None = typer.Option(None, "--manifest")):
    paths = (
        [manifest] if manifest else sorted(Path(".templateparity/model-manifests").glob("*.json"))
    )
    if not paths:
        typer.echo(
            "No model manifests registered. Prepare artifacts explicitly, then run templateparity models register --manifest PATH."
        )
        return
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            typer.echo(f"{path}: unreadable manifest: {exc}", err=True)
            continue
        typer.echo(
            json.dumps(
                {
                    "manifest": str(path),
                    "source_model_id": data.get("source_model_id"),
                    "source_revision": data.get("source_revision"),
                    "mlx_path": data.get("mlx_path"),
                    "mlx_exists": Path(data.get("mlx_path", "")).exists(),
                    "gguf_path": data.get("gguf_path"),
                    "gguf_exists": Path(data.get("gguf_path", "")).exists(),
                    "architecture": data.get("architecture"),
                    "tokenizer_metadata": data.get("tokenizer_metadata"),
                },
                indent=2,
            )
        )


@app.command()
def compare(
    fixture: str = typer.Option(...),
    mode: str = typer.Option("native"),
    mlx_model: Path = typer.Option(...),
    llama_url: str = typer.Option("http://127.0.0.1:8080"),
    max_tokens: int = typer.Option(128, min=1),
    repeat: int = typer.Option(1, min=1, max=10),
    no_persist_raw: bool = typer.Option(False, "--no-persist-raw"),
    output_dir: Path = typer.Option(Path(".templateparity/runs")),
    json_output: bool = typer.Option(False, "--json"),
):
    if mode not in {"native", "shared", "crossover"}:
        raise typer.BadParameter("mode must be native, shared, or crossover")
    conversation = load_fixture(fixture)
    try:
        llama = LlamaCppAdapter(llama_url)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    readiness = llama.probe()
    if not readiness["ready"]:
        typer.echo(readiness["error"], err=True)
        raise typer.Exit(2)
    if not mlx_model.is_dir():
        typer.echo(f"The MLX model directory does not exist: {mlx_model}", err=True)
        raise typer.Exit(2)
    try:
        mlx = MLXAdapter(mlx_model)
        prompts = {"mlx": mlx.render_chat(conversation), "llama": llama.render_chat(conversation)}
    except Exception as exc:
        typer.echo(f"Prompt capture failed: {exc}", err=True)
        raise typer.Exit(2) from exc

    try:
        compatibility, probes_left, probes_right = _tokenizer_checks(mlx, llama, prompts)
    except Exception as exc:
        typer.echo(f"Tokenizer compatibility probes failed: {exc}", err=True)
        raise typer.Exit(2) from exc

    token_comparison = compare_prompts(
        prompts["mlx"], prompts["llama"], compatibility.token_ids_comparable
    )
    generations = {}
    findings = []
    uncertainty = list(compatibility.warnings)
    if mode == "native":
        for runtime_name, adapter, prompt in (
            ("mlx", mlx, prompts["mlx"]),
            ("llama", llama, prompts["llama"]),
        ):
            for run_index in range(1, repeat + 1):
                key = f"{runtime_name}_run_{run_index}"
                generations[key] = adapter.generate_from_tokens(prompt.token_ids, max_tokens)
        if repeat > 1:
            for runtime_name in ("mlx", "llama"):
                outputs = {
                    generations[f"{runtime_name}_run_{i}"].output_text for i in range(1, repeat + 1)
                }
                if len(outputs) > 1:
                    findings.append(
                        Finding(
                            code="TP016",
                            severity="warning",
                            category="repeatability",
                            evidence=f"{runtime_name} produced {len(outputs)} distinct outputs over {repeat} repeated runs.",
                            interpretation="Within-configuration behavior was not stable across these runs.",
                            suggested_action="Inspect effective settings and runtime state before drawing behavioral conclusions.",
                        )
                    )
                    uncertainty.append(f"{runtime_name} repeated output was unstable.")
        findings.extend(
            diagnose(
                prompts["mlx"],
                prompts["llama"],
                generations["mlx_run_1"].stop_configuration,
                generations["llama_run_1"].stop_configuration,
                [generations["mlx_run_1"], generations["llama_run_1"]],
                compatibility.status,
            )
        )
    elif not compatibility.token_ids_comparable:
        detail = "tokenizer compatibility is " + compatibility.status
        _blocked_experiment(mode, prompts, compatibility, output_dir, detail, not no_persist_raw)
    elif mode == "shared":
        left_ids = probes_left["prompt:mlx"]
        right_ids = probes_right["prompt:mlx"]
        if left_ids != right_ids:
            failed = type(compatibility)(
                "mismatch",
                False,
                compatibility.evidence,
                (
                    *compatibility.warnings,
                    "Canonical MLX prompt IDs differ when tokenized by llama.cpp.",
                ),
            )
            _blocked_experiment(
                mode,
                prompts,
                failed,
                output_dir,
                "canonical prompt tokenization differs",
                not no_persist_raw,
            )
        for runtime_name, adapter in (("mlx", mlx), ("llama", llama)):
            for run_index in range(1, repeat + 1):
                generations[f"{runtime_name}_mlx_prompt_run_{run_index}"] = (
                    adapter.generate_from_tokens(left_ids, max_tokens)
                )
        token_comparison["shared_input_token_ids"] = left_ids
        token_comparison["shared_input_source"] = (
            "MLX rendered text, independently tokenized and observed equal by both runtimes"
        )
        if (
            generations["mlx_mlx_prompt_run_1"].output_text
            != generations["llama_mlx_prompt_run_1"].output_text
        ):
            findings.append(
                Finding(
                    code="TP013",
                    severity="warning",
                    category="generation",
                    evidence="Identical checked prompt token IDs produced different output text.",
                    interpretation="Remaining differences may arise from model representation, numerical inference, or stop handling.",
                    suggested_action="Compare model provenance and effective settings; repeat within each runtime.",
                )
            )
    else:
        matrix = {}
        for prompt_name, prompt in prompts.items():
            left_ids = probes_left[f"prompt:{prompt_name}"]
            right_ids = probes_right[f"prompt:{prompt_name}"]
            if left_ids != right_ids:
                matrix[prompt_name] = {
                    "status": "skipped",
                    "reason": "The prompt token IDs differ across runtimes.",
                }
                continue
            matrix[prompt_name] = {"status": "executed", "token_ids": left_ids, "cells": {}}
            for runtime_name, adapter in (("mlx", mlx), ("llama", llama)):
                for run_index in range(1, repeat + 1):
                    key = f"{runtime_name}__{prompt_name}__run_{run_index}"
                    generations[key] = adapter.generate_from_tokens(left_ids, max_tokens)
                matrix[prompt_name]["cells"][runtime_name] = generations[
                    f"{runtime_name}__{prompt_name}__run_1"
                ].output_text
        token_comparison["crossover_matrix"] = matrix
        if not any(cell["status"] == "executed" for cell in matrix.values()):
            _blocked_experiment(
                mode,
                prompts,
                compatibility,
                output_dir,
                "no prompt variant had matching token IDs across both runtimes",
                not no_persist_raw,
            )
        if any(cell["status"] == "skipped" for cell in matrix.values()):
            findings.append(
                Finding(
                    code="TP017",
                    severity="warning",
                    category="compatibility",
                    evidence="One or more crossover prompt variants were skipped because token IDs differed.",
                    interpretation="Only prompt variants with matching checked token IDs can be compared in both runtimes.",
                    suggested_action="Inspect the prompt-specific tokenization evidence.",
                )
            )

    runtime_configs, provenance, mlx_manifest, llama_manifest = _runtime_configs(
        mlx_model, llama, prompts, max_tokens
    )
    if (
        mlx_manifest
        and llama_manifest
        and (mlx_manifest.get("source_model_id"), mlx_manifest.get("source_revision"))
        != (llama_manifest.get("source_model_id"), llama_manifest.get("source_revision"))
    ):
        findings.append(
            Finding(
                code="TP012",
                severity="warning",
                category="provenance",
                evidence="The MLX and GGUF manifests declare different source model revisions.",
                interpretation="Model conversion differences confound cross-runtime behavior attribution.",
                suggested_action="Prepare both artifacts from the same pinned source revision.",
            )
        )
    experiment = Experiment(
        experiment_id=uuid.uuid4().hex[:12],
        mode=mode,
        prompts=prompts,
        runtime_configurations=runtime_configs,
        source_model=provenance,
        generations=generations,
        token_comparison=token_comparison,
        diagnostic_findings=findings,
        compatibility=compatibility.status,
        conclusion=f"{mode.title()} experiment captured. Output differences alone do not establish template causation.",
        uncertainty=[
            *uncertainty,
            "Source-weight identity is unverified because no model registration metadata is configured.",
        ],
    )
    path = _save(experiment, output_dir, not no_persist_raw)
    _emit(
        {
            "experiment_id": experiment.experiment_id,
            "artifact": str(path),
            "mode": mode,
            "compatibility": compatibility.status,
        }
        if json_output
        else f"Saved experiment {experiment.experiment_id} to {path}",
        json_output,
    )


@app.command()
def batch(
    suite: str = typer.Option("core"),
    mlx_model: Path = typer.Option(...),
    llama_url: str = typer.Option("http://127.0.0.1:8080"),
    mode: str = typer.Option("native"),
    max_tokens: int = typer.Option(128, min=1),
    repeat: int = typer.Option(1, min=1, max=10),
    no_persist_raw: bool = typer.Option(False, "--no-persist-raw"),
    output_dir: Path = typer.Option(Path(".templateparity/runs")),
    json_output: bool = typer.Option(False, "--json"),
):
    if suite != "core":
        raise typer.BadParameter("Only the curated core fixture suite is available")
    results = []
    for fixture_name in list_fixtures():
        try:
            compare(
                fixture=fixture_name,
                mode=mode,
                mlx_model=mlx_model,
                llama_url=llama_url,
                max_tokens=max_tokens,
                repeat=repeat,
                no_persist_raw=no_persist_raw,
                output_dir=output_dir,
                json_output=False,
            )
            results.append({"fixture": fixture_name, "status": "completed"})
        except typer.Exit as exc:
            results.append(
                {
                    "fixture": fixture_name,
                    "status": "blocked" if exc.exit_code == 3 else "failed",
                    "exit_code": exc.exit_code,
                }
            )
    _emit(
        results
        if json_output
        else "\n".join(f"{item['fixture']}: {item['status']}" for item in results),
        json_output,
    )


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise typer.BadParameter("The API must bind to localhost.")
    import uvicorn

    uvicorn.run("templateparity.api:app", host=host, port=port)
