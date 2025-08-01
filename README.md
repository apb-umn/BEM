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
| `run_all_results.jl`               | Aggregates baseline, transitions, and optimal tax runs |
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
include("OCModel_opttaxmpi_driver.jl")  
```
---

### 5. Summarize Results
```julia
include("run_all_results.jl")           # For runs all results for the papers
```
---

The code will produce several CSVs that stores the ss moments, transition paths given a reform, and summary of welfare gains for a grid of biz tax rates

## 📚 Citation

If you use this code, please cite the following paper:

> Bhandari, A., Evans, D., & McGrattan, E. (2025). *Approximating Transition Dynamics with Discrete Choice*. University of Minnesota Working Paper.

---
