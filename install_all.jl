# install_all.jl
using Pkg

# List of required packages
packages = [
    "Parameters",
    "LinearAlgebra",
    "BasisMatrices",
    "SparseArrays",
    "Arpack",
    "Roots",
    "KrylovKit",
    "QuantEcon",
    "StatsBase",
    "ForwardDiff",
    "Dierckx",
    "Plots",
    "NPZ",
    "NLsolve",
    "Printf",
    "DataFrames",
    "CSV",
    "Distances",
    "Interpolations",
    "SuiteSparse",
    "TensorOperations",
    "TimerOutputs"
]

# Install any missing packages
for pkg in packages
    if !Pkg.installed(pkg)
        println("Installing $(pkg)...")
        Pkg.add(pkg)
    else
        println("Already installed: $(pkg)")
    end
end

println("\nAll packages are installed and ready to use.")
