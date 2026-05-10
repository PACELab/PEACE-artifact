# Repository Guidelines

## Project Structure & Module Organization
`main.py` orchestrates both stage 1 kernel profiling and stage 2 shared-throughput workflows; keep new entrypoints wired through its flag parser. Core utilities live in `utils/` (`loaddata.py`, `trainer.py`, `predictor.py`) and are shared across scripts in `tests/mps/`, where frequency-scaling and multi-instance experiments reside. Configuration templates are in `configs/` and `template/`, raw assets under `datasets/` and `workloads/`, and generated artifacts belong in `output/`. Legacy integrations remain in `llm-queue/` and `k8sJobs/`; touch them only when a change is intentional.

## Build, Test, and Development Commands
Create a reproducible Python 3.10 environment before running any pipelines:
`conda create -n ml python=3.10 && conda activate ml`
`pip install -r requirements.txt`
Key workflows use `main.py`. Examples:
- `python main.py --train_file <csv> --test_file <csv> -t 100 --train` trains stage 1 models.
- `python main.py --train_file <csv> --test_file <csv> -comb 2 -mt threadclass --predAllacc` drives stage 2 evaluation.
- `python main.py --test_file <csv> --train_file <csv> -m <model.pkl> --sharedThroughputData <path>` performs end-to-end prediction.
Use `bash run_allworkloads.sh` or `bash run_train_splits.sh` to batch complex experiments.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and module-level docstrings in Google style for any public function you add. Use `snake_case` for functions and variables, `CamelCase` for classes, and keep module names lowercase. Prefer pure helpers that return data; push filesystem and WANDB side effects to thin wrappers. Add concise comments only where flags or workflows are non-obvious.

## Testing Guidelines
Tests live alongside scenarios in `tests/`, with MPS-specific suites under `tests/mps/`. Use `pytest` (install if missing) and target focused runs, e.g. `pytest tests/mps/freq_scaling -k stage2`. Seed stochastic code with the existing `-rd/--randomseed` flag and snapshot small CSV fixtures under `tests/mps/.../dataset`. Document any GPU or WANDB prerequisites in the test docstring.

## Commit & Pull Request Guidelines
Recent history favors short summaries (`added freq300 rerun`, `remerge with merge_cudaDL.py`). Keep subject lines under 65 characters, describe *what* changed, and defer rationale to the body when needed. Squash noisy data-only commits before review. PRs should list the affected scripts/flags, attach command logs (train/predict invocations plus key outputs), link tracking issues, and include screenshots or tables when metrics shift. Always note test coverage (`pytest ...`) and any data regeneration steps.

## Data & Secrets
Set `WANDB_API_KEY` and `WANDB_SILENT=true` in your shell before training; never hardcode secrets. Large CSVs and generated checkpoints stay out of version control—stage them inside `output/` and add to `.gitignore` if new. When sharing sample data, anonymize workload names unless they originate from `workloads/` templates.
