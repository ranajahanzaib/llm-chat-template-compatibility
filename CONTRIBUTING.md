# Contributing

Contributions are welcome. For bugs or feature requests, open an issue with a clear description and reproduction steps when applicable. For code changes, open a pull request describing the behavior changed and the checks run.

```bash
git clone https://github.com/ranajahanzaib/llm-chat-template-compatibility.git
cd llm-chat-template-compatibility
uv sync --locked --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Frontend changes can be checked with:

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm typecheck
pnpm build
```

Keep changes focused, include regression tests for bug fixes, and update documentation when commands or user-visible behavior changes. Do not include model weights, experiment artifacts containing private prompts, credentials, or generated build output in pull requests.
