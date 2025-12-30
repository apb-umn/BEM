# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements methods from Bhandari, Evans, & McGrattan (2025) "Approximating Transition Dynamics with Discrete Choice" (JPE Macro). It solves occupational choice models with heterogeneous agents, computing steady states and transition dynamics after tax policy reforms.

## Build and Run Commands

**Install dependencies:**
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

**Compute steady state:**
```bash
julia --project=. -e 'include("OCModelEGM_driver.jl")'
```

**Run transitions:**
```bash
julia --project=. -e 'include("run_all_transition.jl")'
```

**Optimal tax grid search:**
```bash
julia --project=. -e 'include("run_all_opt.jl")'
```

**Full paper outputs:**
```bash
julia --project=. -e 'include("run_all_results.jl"); include("make_data_for_draft.jl")'
```

## Architecture

### Core Model Structure

The model has three business sectors (nonfinancial corporate, financial corporate, private/entrepreneurial) plus workers who make occupational choices between entrepreneurship and paid employment.

**Key data structure:** `OCModel` (defined in `OCModelEGMInputs.jl`) — a mutable struct containing:
- Preference parameters (σ, β, γ)
- Tax rates (τb, τw, τp, τd, τc)
- Productivity shock grids and transition matrices (θb for entrepreneurs, θw for workers)
- Asset grids and basis matrices for spline interpolation
- Equilibrium objects (r, w, tr, value function coefficients, distribution ω)

### Module Hierarchy

```
ZerothOrderApproximation.jl  ← Base objects (Nums, Inputs structs)
        ↓
FirstOrderApproximation.jl   ← First-order perturbation methods
        ↓
SecondOrderApproximation.jl  ← Second-order perturbation methods
        ↓
OCModelEGM_transition.jl     ← Transition FOCs (Fw for workers, Fb for business owners)
```

### Solution Method

1. **Steady State (EGM):** `OCModelEGM.jl` solves household problems using Endogenous Grid Method
   - `setup!()` initializes grids and guesses
   - `policyw()` / `policyb()` compute worker/entrepreneur policy functions
   - `probw()` computes occupational choice probabilities (logistic)

2. **Transition Dynamics:** Perturbation approximations track:
   - Intensive margin: small changes conditional on discrete choices
   - Extensive margin: large changes from occupation switching

3. **Optimal Tax:** `OCModelEGM_opttaxmpi.jl` uses MPI for parallel grid search over business tax rates

### Approximation Objects

- `ZerothOrderApproximation`: Steady-state objects and dimensions
- `FirstOrderApproximation`: Contains `ZO` plus first-order derivatives and IRF storage
- `SecondOrderApproximation`: Contains `FO` plus second-order terms

### Key Equilibrium Conditions

In `OCModelEGM_transition.jl`:
- `Fw()`: Worker FOCs (budget, Euler, envelope, Bellman)
- `Fb()`: Business owner FOCs (includes capital/labor choices, collateral constraints)

## Coding Conventions

- 4-space indentation, no hard tabs
- CamelCase for module files (e.g., `OCModelEGMHighRisk_driver.jl`)
- CSV outputs use consistent prefixes: `data_opt_*`, `SSmoments*`, `Figure*.csv`
- Commit messages: short, lowercase, imperative style

## Testing

No dedicated test suite exists. If adding tests, use Julia's `Test` module and place in `test/runtests.jl`.

## Notes

- Requires Julia 1.12+
- Uses distributed computing for expensive jobs—adjust worker count based on architecture
- Avoid committing large intermediate outputs unless required for paper results
