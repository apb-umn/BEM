@everywhere begin
    include("OCModelE.jl")
    include("OCModelE_transition.jl")

    function setup_old_steady_state()
        OCM = OCModel()
        OCM.χ=1.5
        setup!(OCM)
        OCM.r = 0.04177508910215605
        OCM.tr = 0.6523700237563999
        OCM.ibise = 0
        solvess!(OCM)
        updatecutoffs!(OCM)
        inputs = construct_inputs(OCM)
        X̄_0 = [getX(OCM); OCM.τb]
        A_0 = X̄_0[inputs.Xlab .== :A][1]
        Taub_0 = X̄_0[inputs.Xlab .== :Taub][1]
        ω̄_0_base = sum(reshape(OCM.ω, :, 2), dims=2)
        return OCM, inputs, X̄_0, A_0, Taub_0, ω̄_0_base
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
        taub_val = OCM_new.τb
        println("→ [τb = $taub_val] Setting up transition analysis...")
        println("→ [τb = $taub_val] Constructing inputs...")
        inputs = construct_inputs(OCM_new)
        println("→ [τb = $taub_val] ...done")

        println("→ [τb = $taub_val] Zeroth-order approximation...")
        ZO = ZerothOrderApproximation(inputs)
        println("→ [τb = $taub_val] ...done")

        println("→ [τb = $taub_val] Computing derivatives...")
        computeDerivativesF!(ZO, inputs)
        computeDerivativesG!(ZO, inputs)
        println("→ [τb = $taub_val] ...done")

        println("→ [τb = $taub_val] First-order approximation...")
        FO = FirstOrderApproximation(ZO, OCM_new.T)
        println("→ [τb = $taub_val] ...done")

        println("→ [τb = $taub_val] Computing x, M, L, Js components...")
        compute_f_matrices!(FO)
        compute_Lemma3!(FO)
        compute_Lemma4!(FO)
        compute_Corollary2!(FO)
        compute_Proposition1!(FO)
        compute_BB!(FO)
        println("→ [τb = $taub_val] ...done")

        println("→ [τb = $taub_val] Constructing initial ω̄ vector...")
        ω̄ = reshape(OCM_new.ω, :, 2)
        p̄ = ω̄ ./ sum(ω̄, dims=2)
        p̄[isnan.(p̄[:, 1]), 1] .= 1.0
        p̄[isnan.(p̄[:, 2]), 2] .= 0.0
        ω̄_0 = (p̄ .* ω̄_0_base)[:]
        println("→ [τb = $taub_val] ...done")

        println("→ [τb = $taub_val] Setting initial conditions...")
        FO.X_0 = [A_0; Taub_0] - ZO.P * ZO.X̄
        FO.Θ_0 = [0.0]
        FO.Δ_0 = ω̄_0 - ZO.ω̄
        println("→ [τb = $taub_val] ...done")

        println("→ [τb = $taub_val] Solving transition path...")
        solve_Xt!(FO)
        println("→ [τb = $taub_val] ...done")

        compute_x̂t_ω̂t!(FO)

        println("→ [τb = $taub_val] Computing SO transition path...")
        SO = SecondOrderApproximation(FO=FO)
        SO.X_02 = FO.X_0
        SO.Θ_02 = FO.Θ_0
        SO.ω̂k = FO.ω̂t
        SO.ω̂ak = FO.ω̂at
        SO.x̂k = FO.x̂t
        SO.ŷk = FO.ŷt
        SO.κ̂k = FO.κ̂t
        SO.X̂k = FO.X̂t
        compute_Lemma2_ZZ!(SO)
        compute_lemma3_components!(SO)
        compute_ŷtk!(SO)
        compute_lemma3_ZZ!(SO)
        compute_lemma3_ZZ_kink!(SO)
        compute_Lemma4_ZZ!(SO)
        construct_Laa!(SO)
        compute_Corollary2_ZZ!(SO)
        compute_XZZ!(SO)
        println("→ [τb = $taub_val] ...done")

        println("→ [τb = $taub_val] Constructing X paths and value function...")
        XpathFO = [X̄_0 ZO.X̄ .+ FO.X̂t]
        XpathSO = [X̄_0 ZO.X̄ .+ FO.X̂t .+ 0.5 * SO.X̂tk]
        VinitFO = XpathFO[inputs.Xlab .== :V, 2][1]
        VinitSO = XpathSO[inputs.Xlab .== :V, 2][1]
        println("→ [τb = $taub_val] ...done ✅")
        GC.gc()  # free memory after heavy computation

        return XpathSO, inputs, VinitFO, VinitSO
    end

    function run_grid_point(τb_val::Float64, ρ_τ_vals::Vector{Float64},
                            OCM_old, inputs_old, X̄_old, A_old, Taub_old, ω̄_old)
        local_results = Vector{NamedTuple{(:τb, :ρ_τ, :Vss, :VinitFO, :VinitSO), Tuple{Float64, Float64, Float64, Float64, Float64}}}()
        for ρ_τ_val in ρ_τ_vals
            try
                OCM_new, Xss = setup_new_steady_state(τb_val, OCM_old.τw, ρ_τ_val, OCM_old)
                Vss = Xss[inputs_old.Xlab .== :V][1]
                _, _, VinitFO, VinitSO = perform_transition_analysis(X̄_old, A_old, Taub_old, ω̄_old, OCM_new)
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
        return local_results
        GC.gc()  # free memory after heavy computation

    end
end
