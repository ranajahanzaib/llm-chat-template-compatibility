from __future__ import annotations

import re
from pathlib import Path

from .models import Experiment

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def redact_experiment(experiment: Experiment) -> Experiment:
    """Remove prompt, output, token and content-derived data before persisting."""
    data = experiment.model_dump(mode="json")
    for prompt in data["prompts"].values():
        prompt["rendered_text"] = "[redacted]"
        prompt["rendered_utf8_hash"] = "redacted"
        prompt["rendered_bytes_length"] = 0
        prompt["token_ids"] = []
        prompt["token_count"] = 0
        prompt["token_pieces"] = []
        prompt["special_token_positions"] = []
        prompt["template_options"] = {}
        prompt["template_source"] = "[redacted]"
        prompt["tool_names"] = []
    for generation in data["generations"].values():
        generation["output_text"] = "[redacted]"
        generation["supplied_prompt_token_ids"] = []
        generation["generated_token_ids"] = []
        generation["generated_token_count"] = 0
        generation["stop_configuration"] = {"redacted": True}
    data["token_comparison"] = {"redacted": True}
    data["diagnostic_findings"] = []
    data["conclusion"] = "Content redacted by the persistence privacy setting."
    data["uncertainty"].append(
        "Prompt/output content and content-derived comparisons were redacted before persistence."
    )
    return Experiment.model_validate(data)


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def save(self, experiment: Experiment, persist_raw: bool = True) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(experiment.experiment_id)
        persisted = experiment if persist_raw else redact_experiment(experiment)
        path.write_text(persisted.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def load(self, experiment_id: str) -> Experiment:
        return Experiment.model_validate_json(self._path(experiment_id).read_text(encoding="utf-8"))

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json") if SAFE_ID.fullmatch(p.stem))

    def _path(self, value: str) -> Path:
        if not SAFE_ID.fullmatch(value):
            raise ValueError("Invalid experiment identifier")
        path = (self.root / f"{value}.json").resolve()
        if path.parent != self.root:
            raise ValueError("Invalid experiment path")
        return path
