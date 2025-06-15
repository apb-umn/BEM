@everywhere begin
    include("OCModelE.jl")
    include("OCModelE_transition.jl")

    # --- Define struct first ---
    struct SharedState
        OCM_old
        inputs_old
        X̄_old
        A_old
        Taub_old
        ω̄_old
        ρ_τ_vals
    end

    # --- Setup and Transition Functions ---
    function setup_old_steady_state!(OCM)
        OCM.ibise = 0 
        OCM.iprint = 0
        solvess!(OCM)
        updatecutoffs!(OCM)
        inputs_0 = construct_inputs(OCM)
        X̄_0 = [getX(OCM); OCM.τb]
        A_0 = X̄_0[inputs_0.Xlab .== :A][1]
        Taub_0 = X̄_0[inputs_0.Xlab .== :Taub][1]
        ω̄_0_base = sum(reshape(OCM.ω, :, 2), dims=2)
        ZO_0 = ZerothOrderApproximation(inputs_0)
        Ix̄_0 = ZO_0.x̄ * ZO_0.Φ * ZO_0.ω̄
        return inputs_0, X̄_0, Ix̄_0, A_0, Taub_0, ω̄_0_base
    end

    function setup_new_steady_state(τb, τw, ρ_τ, OCM_old)
        OCM = deepcopy(OCM_old)
        OCM.τb = τb
        OCM.τw = τw
        OCM.ρ_τ = ρ_τ
        OCM.iprint = 0
        OCM.ibise = 0
        ss, lev, shr, res = solvess!(OCM)
        updatecutoffs!(OCM)
        Xss = [getX(OCM); OCM.τb]
        return OCM, Xss
    end

    function perform_transition_analysis(X̄_0, A_0, Taub_0, ω̄_0_base, OCM_new)
        # ... [your existing code unchanged] ...
        # return XpathSO, inputs, VinitFO, VinitSO
    end

    # --- Main function to run grid point ---
    function run_grid_point(τb_val, state::SharedState)
        OCM_old   = state.OCM_old
        inputs    = state.inputs_old
        X̄         = state.X̄_old
        A         = state.A_old
        Taub      = state.Taub_old
        ω̄        = state.ω̄_old
        ρ_τ_vals  = state.ρ_τ_vals

        local_results = Vector{NamedTuple{(:τb, :ρ_τ, :Vss, :VinitFO, :VinitSO), Tuple{Float64, Float64, Float64, Float64, Float64}}}()
        
        for ρ_τ_val in ρ_τ_vals
            try
                OCM_new, Xss = setup_new_steady_state(τb_val, OCM_old.τw, ρ_τ_val, OCM_old)
                Vss = Xss[inputs.Xlab .== :V][1]
                _, _, VinitFO, VinitSO = perform_transition_analysis(X̄, A, Taub, ω̄, OCM_new)
                push!(local_results, (
                    τb = τb_val,
                    ρ_τ = ρ_τ_val,
                    Vss = Vss,
                    VinitFO = VinitFO[1],
                    VinitSO = VinitSO[1]
                ))
            catch e_inner
                @warn "Inner failure at (τb=$τb_val, ρ_τ=$ρ_τ_val)" exception=(e_inner, catch_backtrace())
            end
        end

        GC.gc()
        return local_results
    end
end
