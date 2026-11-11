from templateparity.cli import _save
from templateparity.experiments import synthetic_demo
from templateparity.storage import ArtifactStore


def test_raw_prompt_and_output_can_be_redacted_before_persistence(tmp_path):
    store = ArtifactStore(tmp_path)
    experiment = synthetic_demo()
    _save(experiment, tmp_path, persist_raw=False)
    loaded = store.load("synthetic-demo")
    assert loaded.prompts["mlx"].rendered_text == "[redacted]"
    assert loaded.generations["mlx"].output_text == "[redacted]"
    assert loaded.prompts["mlx"].token_ids == []
    markdown = (tmp_path / "synthetic-demo.md").read_text()
    assert "Reply READY" not in markdown
    assert "Ready." not in markdown
