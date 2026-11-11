import json

import httpx

from templateparity.models import Conversation
from templateparity.runtimes.llama_adapter import LlamaCppAdapter


def test_llama_adapter_uses_numeric_tokens_and_explicit_special_flags():
    calls = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "chat_template": "template",
                    "bos_token": "<BOS>",
                    "eos_token": "<EOS>",
                    "model_path": "/models/model.gguf",
                },
            )
        if request.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<BOS>user: hello"})
        if request.url.path == "/tokenize":
            ids = {"<BOS>": [1], "<EOS>": [2], "<BOS>user: hello": [1, 10, 11]}.get(
                body["content"], [20]
            )
            return httpx.Response(
                200,
                json={
                    "tokens": [{"id": i, "piece": str(i)} for i in ids]
                    if body["with_pieces"]
                    else ids
                },
            )
        if request.url.path == "/completion":
            return httpx.Response(
                200,
                json={
                    "content": "ok",
                    "tokens": [30],
                    "stop_type": "limit",
                    "generation_settings": {"n_predict": 8, "temperature": 0, "top_k": 1},
                },
            )
        return httpx.Response(404)

    adapter = LlamaCppAdapter()
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    assert adapter.probe()["ready"]
    prompt = adapter.render_chat(
        Conversation(conversation_id="x", messages=[{"role": "user", "content": "hello"}])
    )
    assert prompt.token_ids == [1, 10, 11]
    assert prompt.tokenizer_settings["add_special"] is False
    generated = adapter.generate_from_tokens([1, 10, 11], max_tokens=8)
    assert generated.supplied_prompt_token_ids == [1, 10, 11]
    assert generated.generated_token_ids == [30]
    request = next(body for method, path, body in calls if path == "/completion")
    assert request["prompt"] == [1, 10, 11]
    assert request["cache_prompt"] is False and request["return_tokens"] is True


def test_llama_adapter_reports_unexpected_response_shape():
    def handler(request):
        return httpx.Response(200, json={"wrong": "shape"})

    adapter = LlamaCppAdapter()
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    adapter.props = {}
    import pytest

    with pytest.raises(TypeError, match="apply-template"):
        adapter.render_chat(
            Conversation(conversation_id="x", messages=[{"role": "user", "content": "hi"}])
        )
