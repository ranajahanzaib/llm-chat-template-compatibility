from templateparity.comparison import align_tokens
from templateparity.experiments import synthetic_demo
from templateparity.fixtures.loader import list_fixtures, load_fixture
from templateparity.reporting import markdown_report
from templateparity.storage import ArtifactStore


def test_fixtures_load():
    assert len(list_fixtures()) >= 8
    assert load_fixture("simple-user").messages[0].role == "user"


def test_alignment_after_insertion():
    d = align_tokens([1, 2, 3], [1, 9, 2, 3])
    assert d["operations"][1]["tag"] == "insert"
    assert d["common_prefix"] == 1
    assert d["length_delta"] == 1


def test_demo_report_roundtrip(tmp_path):
    e = synthetic_demo()
    store = ArtifactStore(tmp_path)
    store.save(e)
    loaded = store.load(e.experiment_id)
    assert loaded.synthetic and "Synthetic demonstration" in markdown_report(loaded)


def test_safe_ids(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        ArtifactStore(tmp_path).load("../outside")


def test_alignment_reports_deletion_and_replacement():
    from templateparity.comparison import align_tokens

    assert any(op["tag"] == "delete" for op in align_tokens([1, 2, 3], [1, 3])["operations"])
    assert any(op["tag"] == "replace" for op in align_tokens([1, 2], [1, 9])["operations"])
