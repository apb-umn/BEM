# Approximating Transition Dynamics with Discrete Choice

**Anmol Bhandari (University of Minnesota), David Evans (University of Oregon), Ellen McGrattan (University of Minnesota)**

This repository contains the code for the paper:

> Bhandari, A., Evans, D., & McGrattan, E. (2025). "Approximating Transition Dynamics with Discrete Choice." *JPE Macro*, Special Issue on Economic Dynamics, Uncertainty, and Computation.

## Abstract

This paper develops a method for analyzing policy reforms in general equilibrium settings with discrete choice. Computing transition paths in these settings is computationally challenging, particularly in models with substantial heterogeneity and many endogenous states. We extend perturbation methods to handle discrete choice by appropriately tracking both intensive-margin changes conditional on discrete choices that are relatively small and extensive-margin changes resulting from a switch in a discrete choice that are relatively large. The method is fast, scalable, and efficient, providing good initial estimates for global solution methods. We demonstrate our method by analyzing optimal business taxation in a model with occupational choice between entrepreneurship and paid employment.

## Repository Overview

This codebase supports:

- Solving steady-state equilibria using an Endogenous Grid Method (EGM)
- Simulating transition dynamics after tax policy shocks
- Performing optimal tax computations over grids of business tax rates (τ_b)
- Second-, first-, and zeroth-order approximations of dynamic paths

## Key Files and Modules

| Module / Script                | Purpose                                          |
|--------------------------------|--------------------------------------------------|
| `OCModelEGMInputs.jl`          | Defines `OCModel` struct, grids, calibration parameters |
| `OCModelEGM.jl`                | Solves the model's steady state using EGM        |
| `OCModelEGM_driver.jl`         | Driver for computing the baseline equilibrium    |
| `OCModelEGM_transition.jl`     | Transition system logic and FOC residuals        |
| `OCModelEGM_opttaxmpi.jl`      | Parallel grid search for optimal business taxes  |
| `run_all_results.jl`           | Driver for all results in the paper              |
| `SecondOrderApproximation.jl`  | Second-order transition approximation method     |
| `FirstOrderApproximation.jl`   | First-order transition approximation method      |
| `ZerothOrderApproximation.jl`  | Zeroth-order objects                             |
| `utilities.jl`                 | High-dimensional linear algebra utilities        |

## How to Run

### 1. Install Dependencies

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### 2. Compute Steady State

```julia
include("OCModelEGM_driver.jl")
```

### 3. Simulate Transition

```julia
include("run_all_transition.jl")
```

### 4. Compute Optimal Tax

```julia
include("run_all_opt.jl")
```

The code uses Julia's distributed computing to evaluate candidate tax rates in parallel. Adjust the parameters depending on your architecture.

### 5. Generate All Results

```julia
include("run_all_results.jl")
include("make_data_for_draft.jl")
```

The code produces several CSV files containing steady-state moments, transition paths for a given reform, and a summary of welfare gains for a grid of business tax rates.

## Citation

If you use this code, please cite:

> Bhandari, A., Evans, D., & McGrattan, E. (2025). *Approximating Transition Dynamics with Discrete Choice*. JPE Macro, Special Issue on Economic Dynamics, Uncertainty, and Computation.