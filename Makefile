.PHONY: test lint format frontend

test:
	pytest
lint:
	ruff check .
format:
	ruff format .
frontend:
	cd frontend && pnpm typecheck && pnpm build
