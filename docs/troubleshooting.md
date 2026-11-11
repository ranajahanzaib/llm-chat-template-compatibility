# Troubleshooting

- **llama-server unreachable:** start the server on loopback port 8080, or pass `--llama-url`.
- **Unexpected endpoint shape:** check the llama.cpp build's `/props`, `/apply-template`, `/tokenize`, and `/completion` support; adapter errors are surfaced rather than approximated.
- **MLX unavailable:** use Apple Silicon and install `llm-chat-template-compatibility[mlx]` into the active environment.
- **Fixture not found:** run `templateparity fixtures list` to see the bundled fixture names.
- **Shared/crossover skipped:** the compatibility preflight requires matching vocabulary size, BOS/EOS IDs, and checked token vectors. If the server does not expose enough metadata, use native comparison or a server build with the required metadata.
- **Private input in artifacts:** experiment JSON and Markdown include prompt/output text by default; store them in a protected local directory.
