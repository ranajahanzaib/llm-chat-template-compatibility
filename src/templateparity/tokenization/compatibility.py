from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompatibilityResult:
    status: str
    token_ids_comparable: bool
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]


def assess_tokenizer_compatibility(
    left: dict, right: dict, probes_left: dict[str, list[int]], probes_right: dict[str, list[int]]
) -> CompatibilityResult:
    """Conservatively compare metadata and observed vectors; this never proves full equivalence."""
    required = ("vocab_size", "bos_token_id", "eos_token_id")
    missing = [key for key in required if left.get(key) is None or right.get(key) is None]
    if missing:
        return CompatibilityResult(
            "unverified", False, (), ("Tokenizer metadata unavailable: " + ", ".join(missing),)
        )
    mismatches = [
        f"{key} differs ({left[key]!r} vs {right[key]!r})"
        for key in required
        if left[key] != right[key]
    ]
    if mismatches:
        return CompatibilityResult("mismatch", False, (), tuple(mismatches))
    common = sorted(set(probes_left) & set(probes_right))
    if not common:
        return CompatibilityResult(
            "unverified", False, (), ("No shared tokenizer probe vectors were checked.",)
        )
    differing = [name for name in common if probes_left[name] != probes_right[name]]
    if differing:
        return CompatibilityResult(
            "mismatch",
            False,
            (),
            tuple(f"Token IDs differ for probe {name!r}." for name in differing),
        )
    evidence = tuple(f"Token IDs match for probe {name!r}." for name in common)
    return CompatibilityResult(
        "compatible_on_checked_vectors",
        True,
        evidence,
        ("Checked vectors do not prove complete tokenizer equivalence.",),
    )
