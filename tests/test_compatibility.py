from templateparity.tokenization import assess_tokenizer_compatibility


def metadata(**overrides):
    result = {"vocab_size": 100, "bos_token_id": 1, "eos_token_id": 2}
    result.update(overrides)
    return result


def test_matching_metadata_and_vectors_only_claim_checked_compatibility():
    result = assess_tokenizer_compatibility(
        metadata(), metadata(), {"plain": [10, 11]}, {"plain": [10, 11]}
    )
    assert result.status == "compatible_on_checked_vectors"
    assert result.token_ids_comparable
    assert "do not prove complete" in result.warnings[0]


def test_tokenizer_metadata_mismatch_blocks_id_reuse():
    result = assess_tokenizer_compatibility(
        metadata(), metadata(eos_token_id=3), {"plain": [10]}, {"plain": [10]}
    )
    assert result.status == "mismatch"
    assert not result.token_ids_comparable


def test_missing_metadata_remains_unverified():
    result = assess_tokenizer_compatibility(
        metadata(), {"bos_token_id": 1, "eos_token_id": 2}, {"plain": [10]}, {"plain": [10]}
    )
    assert result.status == "unverified"
    assert not result.token_ids_comparable


def test_probe_vector_difference_blocks_id_reuse():
    result = assess_tokenizer_compatibility(
        metadata(), metadata(), {"plain": [10]}, {"plain": [11]}
    )
    assert result.status == "mismatch"
