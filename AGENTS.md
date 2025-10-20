# Repository Guidelines

## Project Goals
This project is a playground for experimenting with INT8/INT4 quantization techniques.  It is not intended to be a production-ready library, we are tinkering and learning.

# Planned Steps
- [ ] Load Qwen-1.5B and confirm it works
- [ ] Quantize the model with bitsandbytes & auto-gptq (GPTQ)
- [ ] Benchmark performance of quantized INT8 & INT4 models with original Qwen-1.5B
- [ ] Visualize weight histograms before and after quantization
- [ ] Measure perplexity shifts, and measure memory footprint
- [ ] Focus on visualizing activations and gradients alongside weights


## Project Structure & Module Organization
The repository is currently flat: `main.py` is the CLI entry point for quantization experiments, `check_gpu.py` probes CUDA availability, and `pyproject.toml` with `uv.lock` tracks dependencies. Add reusable utilities under a new `quantz_expedition/` package and keep exploratory notebooks outside the source tree. Tests belong in `tests/`, mirroring modules (`tests/test_main.py`, etc.), and large artifacts should live in an `assets/` folder with lightweight stubs checked in.

## Build, Test, and Development Commands
- `uv sync` installs the environment from the lock file.
- `uv run python main.py` executes the default workflow; switch to your module’s entry point as needed.
- `uv run python check_gpu.py` validates Torch, CUDA visibility, and driver versions ahead of GPU runs.
- `uv run pytest` runs the full test suite; add `-k pattern` to target specific modules.

## Coding Style & Naming Conventions
Target Python 3.11, prefer explicit type hints, and keep functions below ~50 lines. Use 4-space indentation, snake_case for modules, variables, and functions, and PascalCase for classes. Group imports as stdlib, third-party, then local, and run a quick `uv run python -m py_compile` before pushing to catch syntax errors early.

## Testing Guidelines
Adopt pytest for unit and integration coverage, placing GPU-intensive tests behind `@pytest.mark.cuda` so they can be skipped in CPU-only CI. Name test files `test_<module>.py` and individual cases `test_<behavior>` for clarity. Target ≥80% line coverage on new modules and include regression tests alongside every bug fix.

## Commit & Pull Request Guidelines
There is no commit history yet, so start with Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) and keep subject lines ≤72 characters. Each commit should encapsulate one logical change with related tests and docs. Pull requests must explain motivation, list verification steps or sample outputs, link tracking issues, and attach logs or screenshots when behavior changes.

## Security & Configuration Tips
Never commit credentials or model weights; load tokens (e.g., `HF_TOKEN`, `WANDB_API_KEY`) via environment variables. Document new configuration flags in `README.md` with safe defaults in code. Update `.gitignore` to cover `.uv-cache/`, checkpoints, and experimental datasets before opening a PR.
