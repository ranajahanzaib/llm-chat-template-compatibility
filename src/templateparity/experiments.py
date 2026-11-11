from __future__ import annotations

from .comparison import compare_prompts, diagnose
from .models import Experiment, Generation, Prompt


def synthetic_demo() -> Experiment:
    p1 = Prompt(
        source_runtime="MLX-LM",
        template_name="demo template A",
        rendered_text="<|im_start|>user\nReply READY<|im_end|>\n<|im_start|>assistant\n",
        rendered_utf8_hash="synthetic-a",
        rendered_bytes_length=68,
        token_ids=[1, 10, 20, 30],
        token_count=4,
        token_pieces=["<s>", "user", "Reply READY", "assistant"],
        provenance="Synthetic demonstration data; no model was run.",
        tokenizer_settings={"bos_token_id": 1},
    )
    p2 = Prompt(
        source_runtime="llama.cpp",
        template_name="demo template B",
        rendered_text="<s><|im_start|>user\nReply READY<|im_end|>\n<|im_start|>assistant/analysis\n",
        rendered_utf8_hash="synthetic-b",
        rendered_bytes_length=83,
        token_ids=[1, 1, 10, 20, 31],
        token_count=5,
        token_pieces=["<s>", "<s>", "user", "Reply READY", "assistant/analysis"],
        provenance="Synthetic demonstration data; no model was run.",
        tokenizer_settings={"bos_token_id": 1},
    )
    g1 = Generation(
        runtime="MLX-LM",
        prompt_id="demo-a",
        supplied_prompt_token_ids=p1.token_ids,
        effective_generation_settings={
            "normalized": {"max_tokens": 128, "sampling": "greedy"},
            "runtime": {"sampler": "greedy"},
        },
        output_text="READY",
        generated_token_ids=[50],
        generated_token_count=1,
        finish_reason="eos",
        finish_reason_source="synthetic",
        stop_configuration={"eos_token_ids": [2]},
    )
    g2 = Generation(
        runtime="llama.cpp",
        prompt_id="demo-b",
        supplied_prompt_token_ids=p2.token_ids,
        effective_generation_settings={
            "normalized": {"max_tokens": 128, "sampling": "greedy"},
            "runtime": {"temperature": 0},
        },
        output_text="Ready.",
        generated_token_ids=[51],
        generated_token_count=1,
        finish_reason="stop",
        finish_reason_source="synthetic",
        stop_configuration={"stop_type": "word"},
    )
    comp = compare_prompts(p1, p2)
    fs = diagnose(p1, p2, {"eos_token_ids": [2]}, {"stop_type": "word"}, [g1, g2])
    return Experiment(
        experiment_id="synthetic-demo",
        mode="native",
        synthetic=True,
        source_model={"label": "synthetic only"},
        prompts={"mlx": p1, "llama": p2},
        generations={"mlx": g1, "llama": g2},
        token_comparison=comp,
        diagnostic_findings=fs,
        compatibility="synthetic; no tokenizer claim",
        conclusion="The fabricated example shows a prompt, assistant-prefix, BOS, stop-policy, and output difference. It demonstrates report behavior only; it is not measured inference.",
        uncertainty=["Synthetic data does not establish runtime behavior."],
    )
