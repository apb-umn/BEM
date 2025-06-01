using Distributed, CSV, DataFrames

addprocs(10)

@everywhere begin
    path = "/Users/apb/Dropbox/optimal_business_taxation/noncompliance/Approximation Code/"
    cd(path)

    include("OCModelE.jl")               # defines OCModel and solvecase
    include("OCDriverE_transition.jl")   # defines compute_FO_transition_path

    df = CSV.read(path*"grid_results_with_values.csv", DataFrame)

    # --- Old steady state ---
    OCM_ = OCModel()
    r_val,tr_val = 0.04162616198911325, 0.6221974322920143
    assign!(OCM_, r_val, tr_val)
    inputs_ = construct_inputs(OCM_)

    # Extract initial asset distribution and capital
    X̄_ = getX(OCM_)
    A_0 = X̄_[inputs_.Xlab .== :A][1]                       # Initial capital stock
    ω̄_0_base = sum(reshape(OCM_.ω, :, 2), dims=2)         # Distribution over (a, θ)


    # Combined solver + transition evaluator
    function process_case(τb, τw)
        r_val, tr_val=guess_from_csv(τb, τw, df)
        sol = solvecase(τb, τw, r_val, tr_val)

        if isnan(sol.r) || isnan(sol.tr)
            return merge(sol, (; value = NaN))
        end

        try
            Xpath, Vinit = compute_FO_transition_path(sol.τb, sol.τw, sol.r, sol.tr,
                                                      ω̄_0_base, A_0, X̄_; T = 300)
                                                      
            return merge(sol, (; value = Vinit[1]))
        catch e
            @warn "Transition path failed at τb=$(sol.τb), τw=$(sol.τw)" exception=(e, catch_backtrace())
            return merge(sol, (; value = NaN))
        end
    end
end

# Generate grid of policy parameters
τb_vals = range(0.0, stop = 0.7, length = 10)
τw_vals = range(0.0, stop = 0.7, length = 10)
grid = collect(Iterators.product(τb_vals, τw_vals))

# Run in parallel across all workers
results = pmap(x -> process_case(x[1], x[2]), grid)
flat_results = vec(results)

# Save results
results_df = DataFrame(flat_results)
CSV.write("grid_results_with_values.csv", results_df)
