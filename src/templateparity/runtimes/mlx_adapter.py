from __future__ import annotations

import hashlib

from ..models import Generation, Prompt


class MLXAdapter:
    name = "MLX-LM"

    def __init__(self, model_path):
        try:
            from mlx_lm import load
        except ImportError as e:
            raise RuntimeError(
                "MLX-LM is optional and only available on Apple Silicon; install llm-chat-template-compatibility[mlx]."
            ) from e
        self.model, self.tokenizer = load(str(model_path))
        self.model_path = str(model_path)

    def probe(self):
        return {"ready": True, "runtime": self.name, "model_path": self.model_path}

    def render_chat(self, c):
        messages = [m.model_dump(exclude_none=True) for m in c.messages]
        native = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **c.template_options
        )
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **c.template_options
        )
        ids = list(native.tolist() if hasattr(native, "tolist") else native)
        encoded = self.tokenizer.encode(rendered, add_special_tokens=False)
        warnings = []
        if list(encoded) != ids:
            warnings.append(
                "Encoding rendered text with add_special_tokens=False differs from template-native tokenization."
            )
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        return Prompt(
            source_runtime=self.name,
            template_name="active tokenizer chat template",
            template_source="tokenizer.apply_chat_template",
            template_options=c.template_options,
            rendered_text=rendered,
            rendered_utf8_hash=hashlib.sha256(rendered.encode()).hexdigest(),
            rendered_bytes_length=len(rendered.encode()),
            token_ids=ids,
            token_count=len(ids),
            token_pieces=[str(self.tokenizer.convert_ids_to_tokens(i)) for i in ids],
            special_token_positions=[
                index for index, token_id in enumerate(ids) if token_id in special_ids
            ],
            tokenizer_settings={
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "vocab_size": len(self.tokenizer),
                "native_text_encoding_equal": list(encoded) == ids,
            },
            provenance="Captured from MLX-LM tokenizer.apply_chat_template",
            warnings=warnings,
        )

    def tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def generate_from_tokens(self, token_ids, max_tokens=128):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        pieces = []
        generated = []
        last_response = None
        sampler = make_sampler(temp=0.0)
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=mx.array(token_ids),
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            last_response = response
            pieces.append(response.text)
            generated.append(response.token)
        eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if eos_ids is None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            eos_ids = [] if eos_id is None else eos_id if isinstance(eos_id, list) else [eos_id]
        finish = getattr(last_response, "finish_reason", None)
        return Generation(
            runtime=self.name,
            prompt_id="controlled-token-input",
            supplied_prompt_token_ids=token_ids,
            effective_generation_settings={
                "normalized": {"max_tokens": max_tokens, "sampling": "greedy"},
                "runtime": {"max_tokens": max_tokens, "sampler": "make_sampler(temp=0)"},
            },
            output_text="".join(pieces),
            generated_token_ids=generated,
            generated_token_count=len(generated),
            finish_reason=finish,
            finish_reason_source="MLX-LM stream_generate response" if finish else "unavailable",
            stop_configuration={
                "eos_token_ids": list(eos_ids),
                "emitted_terminal_token": generated[-1] if generated else None,
            },
            warnings=[],
        )
