from __future__ import annotations

import hashlib
import ipaddress
from urllib.parse import urlsplit

import httpx

from ..models import Conversation, Generation, Prompt


def validate_local_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    try:
        is_loopback = hostname == "localhost" or ipaddress.ip_address(hostname).is_loopback
        _ = parsed.port
    except ValueError:
        is_loopback = False
    if (
        parsed.scheme != "http"
        or not is_loopback
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "llama.cpp URL must be a loopback HTTP URL without credentials, query, or path"
        )
    return base_url.rstrip("/")


class LlamaCppAdapter:
    name = "llama.cpp"

    def __init__(self, base_url="http://127.0.0.1:8080", timeout=120):
        self.base_url = validate_local_base_url(base_url)
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.props = {}

    def probe(self):
        try:
            health = self.client.get("/health")
            health.raise_for_status()
            p = self.client.get("/props")
            p.raise_for_status()
            self.props = p.json()
            return {"ready": True, "runtime": self.name, "props": self.props}
        except httpx.HTTPError as e:
            return {
                "ready": False,
                "runtime": self.name,
                "error": f"llama-server is unreachable at {self.base_url}: {e}",
            }

    def _post(self, path, payload):
        r = self.client.post(path, json=payload)
        r.raise_for_status()
        return r.json()

    def render_chat(self, c: Conversation) -> Prompt:
        payload = {"messages": [m.model_dump(exclude_none=True) for m in c.messages]}
        if c.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameter_schema,
                    },
                }
                for t in c.tools
            ]
        if c.template_options:
            payload["chat_template_kwargs"] = c.template_options
        rendered = self._post("/apply-template", payload)
        text = rendered.get("prompt") or rendered.get("content")
        if not isinstance(text, str):
            raise TypeError(f"Unexpected /apply-template response fields: {list(rendered)}")
        tok = self._post(
            "/tokenize",
            {"content": text, "add_special": False, "parse_special": True, "with_pieces": True},
        )
        ids = tok.get("tokens", [])
        pieces = tok.get("pieces", [])
        if ids and isinstance(ids[0], dict):
            pieces = [x.get("piece", "") for x in ids]
            ids = [x.get("id") for x in ids]
        bos_ids = self.tokenize(self.props["bos_token"]) if self.props.get("bos_token") else []
        eos_ids = self.tokenize(self.props["eos_token"]) if self.props.get("eos_token") else []
        special_ids = {ids[0] for ids in (bos_ids, eos_ids) if len(ids) == 1}
        warnings = []
        if not self.props.get("bos_token"):
            warnings.append("The server did not expose its BOS token in /props.")
        if not self.props.get("eos_token"):
            warnings.append("The server did not expose its EOS token in /props.")
        warnings.append(
            "llama.cpp /props does not expose vocabulary size; cross-runtime token compatibility remains unverified."
        )
        return Prompt(
            source_runtime=self.name,
            template_name="llama.cpp native",
            template_source=str(self.props.get("chat_template", "metadata")),
            template_options=c.template_options,
            rendered_text=text,
            rendered_utf8_hash=hashlib.sha256(text.encode()).hexdigest(),
            rendered_bytes_length=len(text.encode()),
            token_ids=ids,
            token_count=len(ids),
            token_pieces=[str(x) for x in pieces],
            special_token_positions=[i for i, x in enumerate(ids) if x in special_ids],
            message_roles=[message.role for message in c.messages],
            tool_names=[tool.name for tool in (c.tools or [])],
            tokenizer_settings={
                "add_special": False,
                "parse_special": True,
                "with_pieces": True,
                "vocab_size": None,
                "bos_token_id": bos_ids[0] if len(bos_ids) == 1 else None,
                "eos_token_id": eos_ids[0] if len(eos_ids) == 1 else None,
            },
            provenance="Captured from llama.cpp /apply-template and /tokenize; special IDs inferred from /props token strings via /tokenize",
            warnings=warnings,
        )

    def tokenize(self, text: str) -> list[int]:
        response = self._post(
            "/tokenize",
            {"content": text, "add_special": False, "parse_special": True, "with_pieces": False},
        )
        return [int(token) for token in response.get("tokens", [])]

    def detokenize(self, token_ids: list[int]) -> str:
        response = self._post("/detokenize", {"tokens": token_ids})
        text = response.get("content")
        if not isinstance(text, str):
            raise TypeError("Unexpected /detokenize response; expected a content string")
        return text

    def generate_from_tokens(self, token_ids: list[int], max_tokens=128) -> Generation:
        settings = {
            "n_predict": max_tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "cache_prompt": False,
            "return_tokens": True,
        }
        result = self._post("/completion", {"prompt": token_ids, **settings})
        text = result.get("content", "")
        tokens = result.get("tokens", [])
        out_ids = [t.get("id") if isinstance(t, dict) else t for t in tokens]
        gs = result.get("generation_settings", {})
        prompt_count = result.get("tokens_evaluated")
        if prompt_count is None:
            prompt_count = result.get("prompt_tokens")
        if isinstance(prompt_count, dict):
            prompt_count = prompt_count.get("count")
        timing = result.get("timings", {}) or {}
        timing_metadata = {}
        for source_key, target_key in (
            ("prompt_ms", "prompt_ms"),
            ("predicted_ms", "generation_ms"),
        ):
            value = timing.get(source_key)
            if isinstance(value, (int, float)):
                timing_metadata[target_key] = float(value)
        warnings = []
        if not out_ids:
            warnings.append(
                "llama.cpp did not return generated token IDs despite return_tokens=true."
            )
        changed = {
            key: (value, gs.get(key))
            for key, value in settings.items()
            if key in gs and gs[key] != value
        }
        if changed:
            warnings.append(f"llama.cpp normalized requested generation settings: {changed!r}")
        if isinstance(prompt_count, int) and prompt_count < len(token_ids):
            warnings.append(
                f"llama.cpp reports evaluating {prompt_count} of {len(token_ids)} supplied prompt tokens; context truncation or another input transformation may have occurred."
            )
        if result.get("truncated") is True or result.get("context_truncated") is True:
            warnings.append("llama.cpp reported context truncation.")
        return Generation(
            runtime=self.name,
            prompt_id="controlled-token-input",
            supplied_prompt_token_ids=token_ids,
            effective_generation_settings={
                "normalized": {
                    "max_tokens": gs.get("n_predict", max_tokens),
                    "sampling": "greedy"
                    if gs.get("temperature", settings["temperature"]) == 0
                    and gs.get("top_k", settings["top_k"]) == 1
                    else "non-greedy",
                },
                "requested": settings,
                "runtime": gs,
            },
            output_text=text,
            generated_token_ids=out_ids,
            generated_token_count=len(out_ids),
            finish_reason=result.get("stop_type"),
            finish_reason_source="llama.cpp completion response"
            if result.get("stop_type")
            else "unavailable",
            stop_configuration={
                "stop_type": result.get("stop_type"),
                "stopping_word": result.get("stopping_word"),
                "context_size": self.props.get("default_generation_settings", {}).get("n_ctx"),
            },
            timing_metadata=timing_metadata,
            warnings=warnings,
        )
