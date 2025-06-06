using Distributed, CSV, DataFrames
include("OCModelE.jl")               # defines OCModel and solvecase
path = "/Users/bhandari/Dropbox/optimal_business_taxation/noncompliance/Approximation Code/"
cd(path)
OCM_ = OCModel()
setup!(OCM_)
solvess!(OCM_)
r_val=OCM_.r
tr_val=OCM_.tr
addprocs(10)

@everywhere begin
    include("OCModelE.jl")               # defines OCModel and solvecase
    include("OCModelE_transition.jl")   # defines compute_FO_transition_path
end

@everywhere path = $path
@everywhere r_val = $r_val
@everywhere tr_val = $tr_val

@everywhere begin
    OCM_ = OCModel()
    assign!(OCM_, r_val, tr_val)
    # file = path * "grid_results_with_values.csv"
    # if isfile(file)
    #     df = CSV.read(file, DataFrame)
    # else
    #     @warn "File not found: $file"
    #     df = DataFrame()
    # end
    assign!(OCM_, r_val, tr_val)
    inputs_ = construct_inputs(OCM_)

    # Extract initial asset distribution and capital
    X̄_ = getX(OCM_)
    A_0 = X̄_[inputs_.Xlab .== :A][1]                       # Initial capital stock
    ω̄_0_base = sum(reshape(OCM_.ω, :, 2), dims=2)         # Distribution over (a, θ)

    function savecase(τb, τw,res)
        df_res=DataFrame()
        push!(df_res,res)
        name = string("case_", round(τb, digits=3), "_", round(τw, digits=3), ".csv")
        CSV.write(name, df_res)
    end

    # Combined solver + transition evaluator
    function process_case(τb, τw)
        # if nrow(df) > 0
        #     r_val, tr_val = guess_from_csv(τb, τw, df)
        #     ibise=0
        #     iprint = 1
        # else
        #     @warn "Skipping guess_from_csv: df is empty"
        #    r_val,tr_val = 0.04167393868478763, 0.627060033883048
            ibise=1
            iprint = 1
        #end
        sol = solvecase_mpi(τb, τw, r_val, tr_val,ibise,iprint)

        if isnan(sol.r) || isnan(sol.tr)
            res=merge(sol, (; value = NaN))
            savecase(τb, τw, res)
            return res
        end

        try
            Xpath, Vinit = compute_FO_transition_path(sol.τb, sol.τw, sol.r, sol.tr,
                                                      ω̄_0_base, A_0, X̄_; T = 300)
           
            res=merge(sol, (; value = Vinit[1]))
            savecase(τb, τw, res)                                              
            return res
        catch e
            @warn "Transition path failed at τb=$(sol.τb), τw=$(sol.τw)" exception=(e, catch_backtrace())
            res=merge(sol, (; value = NaN))
            savecase(τb, τw, res)                                              
            return res
        end
    end
end

# Generate grid of policy parameters
τb_vals = range(0.2, stop = 0.7, length = 15)
τw_vals = range(OCM_.τw, stop = OCM_.τw, length = 1)
grid = collect(Iterators.product(τb_vals, τw_vals))

# Run in parallel across all workers
results = pmap(x -> process_case(x[1], x[2]), grid)
flat_results = vec(results)

# Save results
#results_df = DataFrame(flat_results)
#CSV.write("grid_results_with_values.csv", results_df)
