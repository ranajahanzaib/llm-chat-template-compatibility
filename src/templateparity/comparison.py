from __future__ import annotations

import difflib
import re

from .models import Finding, Prompt


def align_tokens(a: list[int], b: list[int]) -> dict:
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops = [
        {
            "tag": tag,
            "a_start": i1,
            "a_end": i2,
            "b_start": j1,
            "b_end": j2,
            "a": a[i1:i2],
            "b": b[j1:j2],
        }
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
    ]
    first = next((op for op in ops if op["tag"] != "equal"), None)
    return {
        "operations": ops,
        "first_divergence": None
        if first is None
        else {"a": first["a_start"], "b": first["b_start"]},
        "common_prefix": _prefix(a, b),
        "common_suffix": _suffix(a, b),
        "length_delta": len(b) - len(a),
        "identical": a == b,
    }


def _prefix(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def _assistant_generation_prefix(text: str) -> str | None:
    patterns = (
        r"<\|im_start\|>assistant[^\n]*(?:\n)?$",
        r"<\|assistant[^|]*\|>$",
        r"(?m)^assistant(?:/[A-Za-z0-9_.-]+)?[^\n]*(?:\n)?$",
        r"### Assistant:?\s*$",
    )
    matches = [re.search(pattern, text) for pattern in patterns]
    found = [match.group(0) for match in matches if match]
    return found[-1] if found else None


def _suffix(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def diagnose(
    a: Prompt, b: Prompt, stop_a=None, stop_b=None, generations=None, compatibility="unverified"
) -> list[Finding]:
    out = []
    if compatibility in {"unverified", "mismatch"}:
        out.append(
            Finding(
                code="TP011",
                severity="warning",
                category="compatibility",
                evidence=f"Cross-runtime tokenizer compatibility status: {compatibility}.",
                interpretation="Numeric token IDs are shown as observations but are not assumed to have matching semantics.",
                suggested_action="Compare tokenizer metadata and representative vectors; do not reuse IDs across runtimes until checked.",
            )
        )
    if a.rendered_text != b.rendered_text:
        out.append(
            Finding(
                code="TP001",
                severity="warning",
                category="rendering",
                evidence="Rendered prompt text differs.",
                interpretation="The runtimes serialize the logical conversation differently.",
                suggested_action="Inspect the text diff and run a shared-input control.",
            )
        )
    if a.token_ids != b.token_ids:
        out.append(
            Finding(
                code="TP002",
                severity="warning",
                category="tokenization",
                evidence=f"Token sequence lengths: {len(a.token_ids)} and {len(b.token_ids)}.",
                interpretation="Observed prompt token sequences differ.",
                suggested_action="Inspect aligned token operations; establish tokenizer compatibility before comparing IDs.",
            )
        )
        bos_a = a.tokenizer_settings.get("bos_token_id")
        bos_b = b.tokenizer_settings.get("bos_token_id")
        has_bos_insertion = any(
            op["tag"] == "insert" and (bos_b is None or bos_b in op["b"])
            for op in align_tokens(a.token_ids, b.token_ids)["operations"]
        )
        has_bos_deletion = any(
            op["tag"] == "delete" and (bos_a is None or bos_a in op["a"])
            for op in align_tokens(a.token_ids, b.token_ids)["operations"]
        )
        if has_bos_insertion or has_bos_deletion:
            out.append(
                Finding(
                    code="TP003",
                    severity="warning",
                    category="tokenization",
                    evidence="An inserted or deleted token in the aligned sequence is identified as BOS by tokenizer metadata.",
                    interpretation="BOS handling differs for this observed input.",
                    suggested_action="Compare native template tokenization with text tokenization and check for duplicate BOS.",
                )
            )
    for prompt in (a, b):
        bos_id = prompt.tokenizer_settings.get("bos_token_id")
        if bos_id is not None and prompt.token_ids[:2] == [bos_id, bos_id]:
            out.append(
                Finding(
                    code="TP004",
                    severity="warning",
                    category="tokenization",
                    evidence=f"{prompt.source_runtime} begins with two consecutive BOS IDs ({bos_id}).",
                    interpretation="The supplied prompt appears to contain a duplicate BOS token.",
                    suggested_action="Inspect template-native tokenization and automatic BOS insertion.",
                )
            )
    if (
        "system" in a.message_roles or "system" in b.message_roles
    ) and a.rendered_text != b.rendered_text:
        out.append(
            Finding(
                code="TP007",
                severity="info",
                category="template",
                evidence="The conversation includes a system message and the rendered prompt text differs.",
                interpretation="System-role placement or wrapping may contribute to the observed prompt difference.",
                suggested_action="Compare the system-message spans in the text diff.",
            )
        )
    if (
        a.tool_names or b.tool_names or "tool" in a.message_roles or "tool" in b.message_roles
    ) and a.rendered_text != b.rendered_text:
        out.append(
            Finding(
                code="TP009",
                severity="info",
                category="tools",
                evidence=f"Tool names or tool-role messages are present ({a.tool_names!r} vs {b.tool_names!r}) and serialized text differs.",
                interpretation="Tool definitions or result messages may be serialized differently.",
                suggested_action="Inspect the tool-related prompt spans; the tool does not execute tools.",
            )
        )
    tail_a = [
        a.token_pieces[index] for index in a.special_token_positions if index < len(a.token_pieces)
    ][-3:]
    tail_b = [
        b.token_pieces[index] for index in b.special_token_positions if index < len(b.token_pieces)
    ][-3:]
    if tail_a != tail_b and (tail_a or tail_b):
        out.append(
            Finding(
                code="TP006",
                severity="warning",
                category="turn-boundaries",
                evidence=f"Captured special-token pieces near the prompt tail differ: {tail_a!r} vs {tail_b!r}.",
                interpretation="Role or turn-boundary markers may differ; their precise semantics depend on each tokenizer.",
                suggested_action="Inspect the token pieces and the model's special-token metadata.",
            )
        )
    if _assistant_generation_prefix(a.rendered_text) != _assistant_generation_prefix(
        b.rendered_text
    ) and (
        _assistant_generation_prefix(a.rendered_text)
        or _assistant_generation_prefix(b.rendered_text)
    ):
        out.append(
            Finding(
                code="TP005",
                severity="info",
                category="template",
                evidence="Assistant header/prefix text is not identical.",
                interpretation="Generation may begin from different assistant prefixes.",
                suggested_action="Inspect the final rendered prompt segment.",
            )
        )
    if a.template_source != b.template_source:
        out.append(
            Finding(
                code="TP010",
                severity="info",
                category="template",
                evidence=f"Template sources differ: {a.template_source!r} vs {b.template_source!r}.",
                interpretation="The runtimes may use distinct template definitions or metadata sources.",
                suggested_action="Record and compare the exact template source or explicit overrides.",
            )
        )
    if stop_a is not None and stop_b is not None and stop_a != stop_b:
        out.append(
            Finding(
                code="TP008",
                severity="warning",
                category="stopping",
                evidence=f"Stop settings differ: {stop_a!r} vs {stop_b!r}.",
                interpretation="Different stopping policies can change observed output.",
                suggested_action="Repeat with explicitly aligned stop controls.",
            )
        )
    if generations and len(generations) >= 2:
        vals = list(generations.values()) if hasattr(generations, "values") else list(generations)
        if vals[0].effective_generation_settings.get(
            "normalized", vals[0].effective_generation_settings
        ) != vals[1].effective_generation_settings.get(
            "normalized", vals[1].effective_generation_settings
        ):
            out.append(
                Finding(
                    code="TP014",
                    severity="warning",
                    category="generation",
                    evidence="Effective generation settings differ between the captured results.",
                    interpretation="Decoding controls are not aligned, so output comparisons include a settings confound.",
                    suggested_action="Align and record effective settings, then repeat the comparison.",
                )
            )
        if (
            vals[0].supplied_prompt_token_ids == vals[1].supplied_prompt_token_ids
            and vals[0].output_text != vals[1].output_text
        ):
            out.append(
                Finding(
                    code="TP013",
                    severity="warning",
                    category="generation",
                    evidence="Identical supplied token IDs produced different output text.",
                    interpretation="Output differs despite a shared observed input; engine, representation, or stopping differences remain.",
                    suggested_action="Compare model provenance and effective settings, then repeat within each runtime.",
                )
            )
    return out


def compare_prompts(a: Prompt, b: Prompt, token_ids_semantically_comparable: bool = False) -> dict:
    return {
        "text_equal": a.rendered_text == b.rendered_text,
        "token_ids_semantically_comparable": token_ids_semantically_comparable,
        "token_diff": align_tokens(a.token_ids, b.token_ids),
        "text_diff": list(
            difflib.unified_diff(
                a.rendered_text.splitlines(),
                b.rendered_text.splitlines(),
                fromfile=a.source_runtime,
                tofile=b.source_runtime,
                lineterm="",
            )
        ),
    }
