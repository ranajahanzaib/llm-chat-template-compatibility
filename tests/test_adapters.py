from templateparity.runtimes.llama_adapter import LlamaCppAdapter


def test_llama_unavailable_probe_is_actionable():
    result = LlamaCppAdapter("http://127.0.0.1:1", timeout=0.1).probe()
    assert result["ready"] is False
    assert "unreachable" in result["error"]


def test_llama_adapter_rejects_remote_or_credentialed_urls():
    import pytest

    with pytest.raises(ValueError, match="loopback"):
        LlamaCppAdapter("https://example.com")
    with pytest.raises(ValueError, match="loopback"):
        LlamaCppAdapter("http://user:secret@127.0.0.1:8080")
