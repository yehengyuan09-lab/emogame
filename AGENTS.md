# Repository Guidelines

## Project Structure & Module Organization

`app.py` is the Streamlit entry point. Core application code is grouped by responsibility: `crawlers/` collects skin and Weibo data, `vlm/` implements the three-level vision-language pipeline, `feature_engineering/` builds model features, and `data/feature_store.py` manages local persistence. API and business-layer placeholders live in `api/` and `business/`. Root tests cover project integrations; `weiboSpider/` and `minimind/` are bundled upstream projects with their own documentation, dependencies, and tests. Architecture notes belong in `docs/`; dated implementation records belong in `progress/`. Treat databases and downloaded images under `data/wzry_skins/` as local artifacts.

## Build, Test, and Development Commands

Use Python 3.12+ and an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the root unit tests with `python3 -m unittest discover -s tests`. Validate the local hero/skin dataset with `python3 test_wzry_skins.py`. For a small crawler smoke test, run `python3 crawlers/wzry_skin_crawler.py --limit 10`; output is written beneath `data/wzry_skins/`. Run embedded-project tests separately, for example `python3 -m unittest discover -s weiboSpider/tests` after installing its requirements.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules, functions, variables, and test methods; use `PascalCase` for classes and `UPPER_CASE` for constants. Add type hints to public interfaces and short docstrings where behavior is not obvious. Keep I/O at module boundaries and prefer small, testable transformations. The root project has no enforced formatter; match surrounding code. `weiboSpider/` documents `flake8` as its style checker.

## Testing Guidelines

Tests use Python's `unittest`, including `IsolatedAsyncioTestCase` for async crawlers. Name files `test_*.py`, classes `*Tests`, and methods `test_<behavior>`. Mock network calls in unit tests; reserve live requests and Ollama calls for explicit smoke tests. Add regression coverage for crawler parsing, retry behavior, schema changes, and feature-store writes.

## Commit & Pull Request Guidelines

Recent history favors concise imperative subjects, commonly Conventional Commit forms such as `feat(vlm): ...`, `feat: ...`, and `fix: ...`. Keep each commit focused and avoid committing secrets, `.env`, downloaded media, databases, or model weights. Pull requests should explain the behavior change, list verification commands, link relevant issues, and include screenshots for Streamlit UI changes. Call out new external services, credentials, migrations, or large-data requirements explicitly.

## Configuration & Network Safety

Keep credentials in an untracked `.env`. Prefer Chinese package mirrors for downloads, `pip`, and Hugging Face assets. If ordinary network access fails, retry through the configured SOCKS5 proxy at `127.0.0.1:7890`.
