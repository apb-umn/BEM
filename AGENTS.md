# Repository Guidelines

## Project Structure & Module Organization
- Julia source lives in the repo root (`*.jl`). Core model logic is in `OCModelEGM.jl` with inputs/config in `OCModelEGMInputs.jl`.
- Driver scripts orchestrate runs: `OCModelEGM_driver.jl` (baseline), `run_all_transition.jl`, `run_all_opt.jl`, and `run_all_results.jl`.
- Approximation methods are split into `SecondOrderApproximation.jl`, `FirstOrderApproximation.jl`, and `ZerothOrderApproximation.jl`.
- Outputs and inputs are stored as flat files in the root (e.g., `data_opt_*.csv`, `SSmoments*.csv`, `*.tex`, `*.pdf`).

## Build, Test, and Development Commands
- Install dependencies (Julia):
  ```julia
  using Pkg
  Pkg.activate(".")
  Pkg.instantiate()
  ```
- Compute steady state: `julia --project=. -e 'include("OCModelEGM_driver.jl")'`
- Run transitions: `julia --project=. -e 'include("run_all_transition.jl")'`
- Optimal tax grid search: `julia --project=. -e 'include("run_all_opt.jl")'`
- Full paper outputs: `julia --project=. -e 'include("run_all_results.jl"); include("make_data_for_draft.jl")'`

## Coding Style & Naming Conventions
- Use standard Julia formatting: 4-space indentation, no hard tabs.
- File names are descriptive and CamelCase for model modules (e.g., `OCModelEGMHighRisk_driver.jl`).
- Keep new data outputs in CSV with consistent prefixes (e.g., `data_opt_*`, `SSmoments_*`).

## Testing Guidelines
- There is no dedicated test suite in this repository.
- If you add tests, prefer Julia’s built-in `Test` and place files under a new `test/` directory (e.g., `test/runtests.jl`).

## Commit & Pull Request Guidelines
- Recent commit messages are short, lowercase, and imperative (e.g., "update readme", "deleted old file"). Follow that style.
- No PR template is present; include a concise summary, list the main scripts touched, and note any generated outputs or figures.

## Notes for Contributors
- This repo is data-heavy; avoid committing large intermediate outputs unless they are required for the paper results.
- When running expensive jobs, document any parameter changes in the commit message or PR description.
