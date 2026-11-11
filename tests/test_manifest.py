import json

from typer.testing import CliRunner

from templateparity.cli import model_app


def test_model_manifest_register_and_inspect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source"
    mlx = tmp_path / "mlx"
    gguf = tmp_path / "model.gguf"
    source.mkdir()
    mlx.mkdir()
    gguf.write_bytes(b"test artifact")
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_model_id": "org/model",
                "source_revision": "a" * 40,
                "source_path": str(source),
                "mlx_path": str(mlx),
                "gguf_path": str(gguf),
            }
        )
    )
    runner = CliRunner()
    result = runner.invoke(model_app, ["register", "--manifest", str(manifest)])
    assert result.exit_code == 0, result.output
    inspected = runner.invoke(model_app, ["inspect"])
    assert inspected.exit_code == 0 and "org/model" in inspected.output


def test_model_registry_rejects_unpinned_revision(tmp_path):
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(json.dumps({"schema_version": "1.0", "source_revision": "main"}))
    result = CliRunner().invoke(model_app, ["register", "--manifest", str(manifest)])
    assert result.exit_code != 0
