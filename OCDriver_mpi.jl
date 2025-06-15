using Distributed, CSV, DataFrames

# Set working directory
cd("/Users/bhandari/Dropbox/optimal_business_taxation/noncompliance/Approximation Code/anm_git/")

# Add workers
addprocs(6)

#
filenamesuffix="lowchi"

# Load all worker-side logic
include("OCModel_mpi.jl")

# Define your parameter grid
τb_vals = collect(range(0.21, stop=0.7, length=36))
ρ_τ_vals = [0.0]

# Run steady state once on main process
OCM_old, inputs_old, X̄_old, A_old, Taub_old, ω̄_old = setup_old_steady_state()

# Closure for parallel run
run_closure = τb_val -> run_grid_point(τb_val, ρ_τ_vals, OCM_old, inputs_old, X̄_old, A_old, Taub_old, ω̄_old)

# Run in parallel
results_all = pmap(run_closure, τb_vals)

# Process results
successes = filter(!isempty, results_all)
failures = filter(isempty, results_all)
println("Successful τb count: ", length(successes))
println("Failed τb count: ", length(failures))

df_results = DataFrame(reduce(vcat, successes))
CSV.write("data_opt_" * filenamesuffix * ".csv", df_results)

# Analyze and filter
analyze_optimal_taub("data_opt_"*filenamesuffix* ".csv",col=:VinitSO)
println("...done ✅")


# drop unsolved cases based on VinitSO/VinitFO
df_results = dropmissing(df_results, :VinitSO)
df_filtered = filter(row -> abs(row.VinitFO - row.VinitSO) ≤ 0.01 * abs(row.VinitSO), df_results)
CSV.write("data_opt_"*filenamesuffix* "_filtered.csv", df_filtered)
analyze_optimal_taub("data_opt_"*filenamesuffix* "_filtered.csv",col=:VinitSO)

# smooth the VinitSO column
df_filtered_smooth = smooth_VinitSO(df_filtered, 0.1)
CSV.write("data_opt_"*filenamesuffix* "_smooth_filtered.csv", df_filtered_smooth)
analyze_optimal_taub("data_opt_"* filenamesuffix *"_smooth_filtered.csv",col=:VinitSO_smooth)
