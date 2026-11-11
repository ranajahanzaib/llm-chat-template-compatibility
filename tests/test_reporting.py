from templateparity.experiments import synthetic_demo
from templateparity.reporting import markdown_report


def test_markdown_report_has_provenance_and_limitations():
    text = markdown_report(synthetic_demo())
    assert "Synthetic demonstration" in text
    assert "Interpretation limits" in text
    assert "Provenance" in text
