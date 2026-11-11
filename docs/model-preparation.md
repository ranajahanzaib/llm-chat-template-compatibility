# Model preparation

Model downloads and conversions are explicit. For comparable investigations, use the same pinned Hugging Face source revision and convert independently to MLX float16 and GGUF F16 where practical. The repository includes `scripts/prepare_models.sh` for this workflow; it requires the Hugging Face CLI, MLX-LM, Python, and a llama.cpp checkout. Invoke it with the repository name, full commit SHA, output directory, and llama.cpp checkout path. Register the generated manifest with `templateparity models register --manifest PATH`.

The manifest records the source revision, conversion commands and tool versions, architecture, tokenizer metadata, special token IDs, dtypes, and file checksums. Never infer shared weights solely from matching model names. Review the upstream MLX-LM and llama.cpp conversion documentation for the versions used in a given run.
