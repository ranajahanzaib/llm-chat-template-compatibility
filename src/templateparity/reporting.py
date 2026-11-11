from .models import Experiment


def markdown_report(e: Experiment) -> str:
    label = (
        "Synthetic demonstration — not measured model inference."
        if e.synthetic
        else "Captured experiment"
    )
    lines = [
        f"# LLM Chat Template Compatibility report: {e.experiment_id}",
        "",
        f"> {label}",
        "",
        f"**Mode:** {e.mode}",
        f"**Compatibility:** {e.compatibility}",
        "",
        "## Conclusion",
        "",
        e.conclusion or "No conclusion recorded.",
        "",
        "## Prompt comparison",
        "",
    ]
    for name, p in e.prompts.items():
        lines += [
            f"### {name} ({p.source_runtime})",
            "",
            f"- Template: {p.template_name}",
            f"- UTF-8 bytes: {p.rendered_bytes_length}",
            f"- Tokens: {p.token_count}",
            f"- Provenance: {p.provenance}",
            "",
            "```text",
            p.rendered_text,
            "```",
            "",
        ]
    if e.token_comparison:
        lines += [
            "## Token alignment",
            "",
            "```json",
            __import__("json").dumps(e.token_comparison, indent=2),
            "```",
            "",
        ]
    lines += ["## Findings", ""]
    lines += [
        f"- **{f.code} ({f.severity})** {f.evidence} {f.interpretation} Suggested: {f.suggested_action}"
        for f in e.diagnostic_findings
    ] or ["No differences detected."]
    lines += [
        "",
        "## Interpretation limits",
        "",
        "Prompt differences are observed evidence, not proof of behavioral causation. Identical token IDs do not guarantee identical output across inference engines.",
    ]
    return "\n".join(lines) + "\n"
