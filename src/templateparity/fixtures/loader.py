from pathlib import Path

import yaml

from ..models import Conversation

ROOT = Path(__file__).resolve().parents[3] / "fixtures"


def load_fixture(name: str) -> Conversation:
    if not name.replace("-", "").isalnum():
        raise ValueError("Invalid fixture name")
    path = ROOT / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown fixture {name!r}; run templateparity fixtures list")
    return Conversation.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def list_fixtures() -> list[str]:
    return sorted(p.stem for p in ROOT.glob("*.yaml"))
