# Approximating Transition Dynamics with Discrete Choice

**Anmol Bhandari (University of Minnesota), David Evans (University of Oregon), Ellen McGrattan (University of Minnesota)**

This repository contains the code for the paper:

> **"Approximating Transition Dynamics with Discrete Choice"**  
> *Anmol Bhandari, David Evans, and Ellen McGrattan*  

## 📄 Abstract

This paper develops a method to analyze policy reforms in environments with discrete choice, such as occupational choice or default. Computing transition paths in these settings is computationally challenging, especially in models with substantial heterogeneity and many endogenous states. We extend perturbation methods to handle discrete choice by tracking both:
- **Intensive-margin** changes conditional on choices (typically smooth and small), and
- **Extensive-margin** changes from switching choices (discrete and potentially large).

The method is fast, scalable, and efficient — providing high-quality initial estimates for global solution methods. We apply this approach to evaluate optimal business taxation in a heterogeneous-agent model with occupational choice between wage work and entrepreneurship.

---

## 🧭 Repository Overview

This codebase supports:
- Solving **steady state equilibria** using an Endogenous Grid Method (EGM).
- Simulating **transition dynamics** after tax policy shocks.
- Performing **optimal tax computations** over grids of business tax rates (`τ_b`).
- **Second- and zeroth-order approximations** of dynamic paths.

---

## 📂 Key Files and Modules

| Module / Script                     | Purpose |
|------------------------------------|---------|
| `OCModelEGMInputs.jl`              | Defines `OCModel` struct, grids, calibration parameters |
| `OCModelEGM.jl`                    | Solves the model’s steady state using EGM |
| `OCModelEGM_driver.jl`             | Driver for computing the baseline equilibrium |
| `OCModelEGM_transition.jl`         | Transition system logic and FOC residuals |
| `OCModelEGM_transition_driver.jl`  | Computes transitions for given tax reforms |
| `OCModelEGM_opttaxmpi.jl`          | Parallel grid search for optimal business taxes |
| `OCModel_opttaxmpi_driver.jl`      | Master script for optimal tax search |
| `run_all_opt_chi.jl`               | Runs optimal tax search under alternative collateral constraints |
| `run_opt_high_risk.jl`             | Optimal tax under high income risk calibration |
| `run_all_results.jl`               | Aggregates baseline, transitions, and optimal tax runs |
| `summarize_opt.jl`                 | Compares and exports baseline vs. optimal equilibria |
| `SecondOrderApproximation.jl`      | 2nd-order transition approximation method |
| `ZerothOrderApproximation.jl`      | Baseline discrete transition approximation |
| `utilities.jl`                     | High-dimensional linear algebra utilities |

---

## 🚀 How to Run

### 1. Install Dependencies
Run the following in Julia:
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
include("OCModelEGM_transition_driver.jl")
```

### 4. Compute Optimal Tax
```julia
include("OCModel_opttaxmpi_driver.jl")  # Base case
include("run_all_opt_chi.jl")           # For χ sensitivity
include("run_opt_high_risk.jl")         # For high-risk calibration
```

### 5. Summarize Results
```julia
include("summarize_opt.jl")
```

This exports:
- Steady state moment tables: `SSmoments*.csv`
- Optimal policy comparisons: `moments_comparison_*.csv`
- Tax policy outputs: `data_opt_*.csv`, filtered and smoothed

---

## 📊 Output Files

- `data_opt_base.csv` — raw results of optimal tax grid
- `data_opt_base_filtered.csv` — filtered by stability
- `data_opt_base_smooth_filtered.csv` — smoothed for reporting
- `transition_comparison_*.pdf` — plots of transition paths
- `moments_comparison_*.csv` — steady state comparisons

---

## 📚 Citation

If you use this code, please cite the following paper:

> Bhandari, A., Evans, D., & McGrattan, E. (2025). *Approximating Transition Dynamics with Discrete Choice*. University of Minnesota Working Paper.


---

## 🧠 Notes for Researchers

- Transition logic combines both continuous perturbation and discrete switching.
- Tax simulations focus on welfare across `Vss`, `VinitSO`, and `VinitFO`.
- Code is modular for extensions: e.g., new frictions, occupations, or policy instruments.
