using Distributed, CSV, DataFrames
path = "/Users/bhandari/Dropbox/optimal_business_taxation/noncompliance/Approximation Code/anm_git/"
cd(path)

addprocs(8)  # or however many cores you want to use

@everywhere begin
    include("OCModelE.jl")               # defines OCModel and solvecase
    include("OCModelE_transition_speed.jl")
    function setup_old_steady_state()
        OCM = OCModel()
        OCM.χ=2.0
        setup!(OCM)
        OCM.r = 0.04176095778425395 #baseline
        OCM.tr = 0.65256444424614563 #baseline
        OCM.ibise = 0
        solvess!(OCM)
        assign!(OCM, OCM.r, OCM.tr)
        inputs = construct_inputs(OCM)
        X̄_0 = [getX(OCM); OCM.τb]
        A_0 = X̄_0[inputs.Xlab .== :A][1]
        Taub_0 = X̄_0[inputs.Xlab .== :Taub][1]
        ω̄_0_base = sum(reshape(OCM.ω, :, 2), dims=2)
        return OCM, inputs, X̄_0,A_0, Taub_0, ω̄_0_base
    end
    
        
    function setup_new_steady_state(τb, τw, ρ_τ, OCM_old)
        OCM = deepcopy(OCM_old)
        OCM.τb = τb
        OCM.τw = τw
        OCM.ρ_τ = ρ_τ
        OCM.iprint = 1
        OCM.ibise = 0
        assign!(OCM, OCM_old.r, OCM_old.tr)
        ss, lev, shr, res = solvess!(OCM)
        Xss = [getX(OCM); OCM.τb]
        return OCM, Xss
    end
    
    function perform_transition_analysis(X̄_0,A_0, Taub_0, ω̄_0_base, OCM_new)
        println("→ Constructing inputs...")
        inputs = construct_inputs(OCM_new)
        println("...done")
    
        println("→ Zeroth-order approximation...")
        ZO = ZerothOrderApproximation(inputs)
        println("...done")
    
        println("→ Computing derivatives...")
        computeDerivativesF!(ZO, inputs)
        computeDerivativesG!(ZO, inputs)
        println("...done")
    
        println("→ First-order approximation...")
        FO = FirstOrderApproximation(ZO, OCM_new.T)
        println("...done")
    
        println("→ Computing x,M,L,Js components...")
        compute_f_matrices!(FO)
        compute_Lemma3!(FO)
        compute_Lemma4!(FO)
        compute_Corollary2!(FO)
        compute_Proposition1!(FO)
        compute_BB!(FO)
        println("...done")
    
        println("→ Constructing initial ω̄ vector...")
        ω̄ = reshape(OCM_new.ω, :, 2)
        p̄ = ω̄ ./ sum(ω̄, dims=2)
        p̄[isnan.(p̄[:, 1]), 1] .= 1.0
        p̄[isnan.(p̄[:, 2]), 2] .= 0.0
        ω̄_0 = (p̄ .* ω̄_0_base)[:]
        println("...done")
    
        println("→ Setting initial conditions...")
        FO.X_0 = [A_0; Taub_0] - ZO.P * ZO.X̄
        FO.Θ_0 = [0.0]
        FO.Δ_0 = ω̄_0 - ZO.ω̄
        println("...done")
    
        println("→ Solving transition path...")
        solve_Xt!(FO)
        println("...done")
    
        println("→ Constructing X paths and value function...")
        Xpath = [X̄_0 ZO.X̄ .+ FO.X̂t]
        Vinit = ZO.X̄[inputs.Xlab .== :V] + FO.X̂t[inputs.Xlab .== :V, 1]
    
    
        println("...done ✅")
    
        return Xpath,inputs, Vinit
    end
end


# Define your parameter grid
τb_vals = collect(range(0.2, stop=0.7, length=24))
ρ_τ_vals = collect(range(0.0, stop=0.0, length=1))

@everywhere function run_grid_point(τb_val, ρ_τ_vals)
    local_results = NamedTuple[]
    try
        OCM_old, inputs_old, X̄_old, A_old, Taub_old, ω̄_old = setup_old_steady_state()
        for ρ_τ_val in ρ_τ_vals
            try
                OCM_new, Xss = setup_new_steady_state(τb_val, OCM_old.τw, ρ_τ_val, OCM_old)
                Vss = Xss[inputs_old.Xlab .== :V][1]
                _, _, Vinit = perform_transition_analysis(X̄_old, A_old, Taub_old, ω̄_old, OCM_new)
                push!(local_results, (τb=τb_val, ρ_τ=ρ_τ_val, Vss=Vss, Vinit=Vinit[1]))
            catch e_inner
                @warn "Inner failure at (τb=$τb_val, ρ_τ=$ρ_τ_val)" exception=(e_inner, catch_backtrace())
            end
        end
    catch e_outer
        @warn "Outer failure at τb = $τb_val" exception=(e_outer, catch_backtrace())
    end
    return local_results
end


# Parallel map over τb_vals
results = pmap(τb -> run_grid_point(τb, ρ_τ_vals), τb_vals)
successes = filter(!isempty, results)
failures = filter(isempty, results)
println("Successful τb count: ", length(successes))
println("Failed τb count: ", length(failures))

# Flatten the list of results
df_results = DataFrame(reduce(vcat, successes))

CSV.write("data_opt_base.csv", df_results)
