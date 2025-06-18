
include("OCModelE_transition.jl") # has the F,G,ff and several helper functions
using Distributed

# Add workers
addprocs(8)

# Load all worker-side logic
include("OCModel_mpi.jl")


#
filenamesuffix="tauw20"
#gval=0.0
#τpval = 0.0
#τdval=0.0
#τcval=0.0
τwval=0.20
#bval=0.0

# Define your parameter grid
τb_vals = collect(range(0.21, stop=0.7, length=36))
ρ_τ_vals = [0.0]

println("Setting up old steady state (takes a few minutes) on master node...")
OCM_old = OCModel()
setup!(OCM_old)
#OCM_old.τp = τpval
#OCM_old.τd = τdval
#OCM_old.τc = τcval
OCM_old.τw = τwval
#OCM_old.b = bval
#OCM_old.g = gval
OCM_old.ibise=1

inputs_old, X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old = setup_old_steady_state!(OCM_old)
println("Old steady state setup complete..")

state = SharedState(OCM_old, inputs_old, X̄_old, A_old, Taub_old, ω̄_0_old, ρ_τ_vals)

for p in workers()
    Distributed.remotecall_eval(Main, p, quote
        global state = deepcopy($state)
    end)
end


# Closure for parallel run
run_closure = τb_val -> run_grid_point(τb_val, state)

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

# Clean up workers
rmprocs(workers())


# Add workers
addprocs(8)


# Load all worker-side logic
include("OCModel_mpi.jl")


#
filenamesuffix="tauw60"
#gval=0.0
#τpval = 0.0
#τdval=0.0
#τcval=0.0
τwval=0.60
#bval=0.0

# Define your parameter grid
τb_vals = collect(range(0.21, stop=0.7, length=36))
ρ_τ_vals = [0.0]

println("Setting up old steady state (takes a few minutes) on master node...")
OCM_old = OCModel()
setup!(OCM_old)
#OCM_old.τp = τpval
#OCM_old.τd = τdval
#OCM_old.τc = τcval
OCM_old.τw = τwval
#OCM_old.b = bval
#OCM_old.g = gval
OCM_old.ibise=1

inputs_old, X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old = setup_old_steady_state!(OCM_old)
println("Old steady state setup complete..")

state = SharedState(OCM_old, inputs_old, X̄_old, A_old, Taub_old, ω̄_0_old, ρ_τ_vals)

for p in workers()
    Distributed.remotecall_eval(Main, p, quote
        global state = deepcopy($state)
    end)
end


# Closure for parallel run
run_closure = τb_val -> run_grid_point(τb_val, state)

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

# Clean up workers
rmprocs(workers())



# Add workers
addprocs(8)


# Load all worker-side logic
include("OCModel_mpi.jl")

#
filenamesuffix="tauw30"
#gval=0.0
#τpval = 0.0
#τdval=0.0
#τcval=0.0
τwval=0.30
#bval=0.0

# Define your parameter grid
τb_vals = collect(range(0.21, stop=0.7, length=36))
ρ_τ_vals = [0.0]

println("Setting up old steady state (takes a few minutes) on master node...")
OCM_old = OCModel()
setup!(OCM_old)
#OCM_old.τp = τpval
#OCM_old.τd = τdval
#OCM_old.τc = τcval
OCM_old.τw = τwval
#OCM_old.b = bval
#OCM_old.g = gval
OCM_old.ibise=1

inputs_old, X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old = setup_old_steady_state!(OCM_old)
println("Old steady state setup complete..")

state = SharedState(OCM_old, inputs_old, X̄_old, A_old, Taub_old, ω̄_0_old, ρ_τ_vals)

for p in workers()
    Distributed.remotecall_eval(Main, p, quote
        global state = deepcopy($state)
    end)
end


# Closure for parallel run
run_closure = τb_val -> run_grid_point(τb_val, state)

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

# Clean up workers
rmprocs(workers())




# Add workers
addprocs(8)


# Load all worker-side logic
include("OCModel_mpi.jl")

#
filenamesuffix="tauw10"
#gval=0.0
#τpval = 0.0
#τdval=0.0
#τcval=0.0
τwval=0.10
#bval=0.0

# Define your parameter grid
τb_vals = collect(range(0.21, stop=0.7, length=36))
ρ_τ_vals = [0.0]

println("Setting up old steady state (takes a few minutes) on master node...")
OCM_old = OCModel()
setup!(OCM_old)
#OCM_old.τp = τpval
#OCM_old.τd = τdval
#OCM_old.τc = τcval
OCM_old.τw = τwval
#OCM_old.b = bval
#OCM_old.g = gval
OCM_old.ibise=1

inputs_old, X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old = setup_old_steady_state!(OCM_old)
println("Old steady state setup complete..")

state = SharedState(OCM_old, inputs_old, X̄_old, A_old, Taub_old, ω̄_0_old, ρ_τ_vals)

for p in workers()
    Distributed.remotecall_eval(Main, p, quote
        global state = deepcopy($state)
    end)
end


# Closure for parallel run
run_closure = τb_val -> run_grid_point(τb_val, state)

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

# Clean up workers
rmprocs(workers())



