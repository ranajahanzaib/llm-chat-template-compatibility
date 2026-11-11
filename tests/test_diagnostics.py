from templateparity.experiments import synthetic_demo


def test_demo_contains_required_observations():
    demo = synthetic_demo()
    codes = {finding.code for finding in demo.diagnostic_findings}
    assert {"TP001", "TP002", "TP005", "TP008", "TP004", "TP011"} <= codes
    assert demo.synthetic is True


def test_synthetic_data_is_explicitly_disclaimed():
    report = synthetic_demo()
    from templateparity.reporting import markdown_report

    assert "Synthetic demonstration — not measured model inference." in markdown_report(report)


def test_system_tool_and_boundary_findings_are_evidence_backed():
    from templateparity.comparison import diagnose
    from templateparity.models import Prompt

    common = {
        "rendered_utf8_hash": "h",
        "rendered_bytes_length": 1,
        "token_count": 2,
        "provenance": "test",
        "message_roles": ["system", "user", "tool"],
        "tool_names": ["lookup"],
        "special_token_positions": [1],
        "tokenizer_settings": {},
    }
    left = Prompt(
        source_runtime="A",
        template_name="a",
        rendered_text="<|system|>A<|tool_end|>",
        token_ids=[1, 2],
        token_pieces=["a", "<|tool_end|>"],
        **common,
    )
    right = Prompt(
        source_runtime="B",
        template_name="b",
        rendered_text="system: A<|end_turn|>",
        token_ids=[3, 4],
        token_pieces=["b", "<|end_turn|>"],
        **common,
    )
    codes = {finding.code for finding in diagnose(left, right)}
    assert {"TP006", "TP007", "TP009"} <= codes
