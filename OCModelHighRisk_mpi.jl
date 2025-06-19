@everywhere begin
    include("OCModelEGMHighRisk.jl")
    include("OCModelEHighRisk_transition.jl")

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
        #OCM.ibise = 0 
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
        try
            taub_val = OCM_new.τb
            println("→ [τb = $taub_val] Setting up transition analysis...")
    
            # Your existing logic here
            inputs = construct_inputs(OCM_new)
            ZO = ZerothOrderApproximation(inputs)
            computeDerivativesF!(ZO, inputs)
            computeDerivativesG!(ZO, inputs)
            FO = FirstOrderApproximation(ZO, OCM_new.T)
            compute_f_matrices!(FO)
            compute_Lemma3!(FO)
            compute_Lemma4!(FO)
            compute_Corollary2!(FO)
            compute_Proposition1!(FO)
            compute_BB!(FO)
    
            ω̄ = reshape(OCM_new.ω, :, 2)
            p̄ = ω̄ ./ sum(ω̄, dims=2)
            p̄[isnan.(p̄[:, 1]), 1] .= 1.0
            p̄[isnan.(p̄[:, 2]), 2] .= 0.0
            ω̄_0 = (p̄ .* ω̄_0_base)[:]
    
            FO.X_0 = [A_0; Taub_0] - ZO.P * ZO.X̄
            FO.Θ_0 = [0.0]
            FO.Δ_0 = ω̄_0 - ZO.ω̄
    
            solve_Xt!(FO)
            compute_x̂t_ω̂t!(FO)
    
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
    
            XpathFO = [X̄_0 ZO.X̄ .+ FO.X̂t]
            XpathSO = [X̄_0 ZO.X̄ .+ FO.X̂t .+ 0.5 * SO.X̂tk]
            VinitFO = XpathFO[inputs.Xlab .== :V, 2][1]
            VinitSO = XpathSO[inputs.Xlab .== :V, 2][1]
    
            GC.gc()
            return XpathSO, inputs, VinitFO, VinitSO
    
        catch e
            @warn "Transition analysis failed for τb = $(OCM_new.τb)" exception=(e, catch_backtrace())
            return nothing
        end
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


    Vss_old = X̄[inputs.Xlab .== :V][1]

    getCE(Vnew)=(Vnew/Vss_old)^(1/(1-OCM_old.σ)) - 1. #check david


    local_results = Vector{NamedTuple{(:τb, :ρ_τ, :Vss, :VinitFO, :VinitSO,:CESS,:CEFO,:CESO), Tuple{Float64, Float64, Float64, Float64, Float64,Float64,Float64,Float64}}}()
    
    for ρ_τ_val in ρ_τ_vals
        try
            OCM_new, Xss = setup_new_steady_state(τb_val, OCM_old.τw, ρ_τ_val, OCM_old)
            Vss = Xss[inputs.Xlab .== :V][1]

            result = perform_transition_analysis(X̄, A, Taub, ω̄, OCM_new)

            if result === nothing
                @warn "perform_transition_analysis returned nothing at τb=$τb_val, ρ_τ=$ρ_τ_val"
                continue
            end

            _, _, VinitFO, VinitSO = result

            CESS = getCE(Vss)
            CEFO= getCE(VinitFO[1])
            CESO= getCE(VinitSO[1])

            push!(local_results, (
                τb = τb_val,
                ρ_τ = ρ_τ_val,
                Vss = Vss,
                VinitFO = VinitFO[1],
                VinitSO = VinitSO[1],
                CESS = CESS,
                CEFO = CEFO,
                CESO = CESO
            ))

        catch e_inner
            @warn "Inner failure at (τb=$τb_val, ρ_τ=$ρ_τ_val)" exception=(e_inner, catch_backtrace())
        end
    end

    GC.gc()
    return local_results
end

end
# --- End of @everywhere block ---
function analyze_optimal_taub(file::String; col::Symbol = :VinitSO)
    # Step 1: Read the data
    df = CSV.read(file, DataFrame)

    # Step 2: Find the best τb using col and Vss
    idx_col = argmax(df[!, col])
    idx_vss = argmax(df.Vss)

    best_taub_col = df.τb[idx_col]
    best_taub_vss = df.τb[idx_vss]

    # Step 3: Construct summary table
    summary = DataFrame(
        Criterion = ["Best by $(col)", "Best by Vss"],
        τb = [best_taub_col, best_taub_vss],
        Vinit = [df[idx_col, col], df[idx_vss, col]],
        Vss = [df.Vss[idx_col], df.Vss[idx_vss]]
    )

    println("\nSummary Table:")
    pretty_table(summary; formatters = ft_printf("%.4f"))

    # Save summary table to LaTeX file
    open("optimal_tau_summary.tex", "w") do io
        pretty_table(
            io, summary;
            header = names(summary),
            backend = Val(:latex),
            tf = tf_latex_booktabs,
            formatters = ft_printf("%.4f")
        )
    end

    # Step 4: Plot smoothed column vs τb with optima marked
    smooth_col_name = Symbol("smooth_", col)
    if hasproperty(df, smooth_col_name)
        smooth_vals = df[!, smooth_col_name]
    else
        # If no smooth version exists, fall back to raw column
        smooth_vals = df[!, col]
    end

    default(linewidth=2)
    plt = plot(df.τb, smooth_vals, label = "Value", xlabel = "τb", ylabel = "V",
               title = "Value vs τb", legend = :bottomright, grid = true)

    vline!([best_taub_col], label = "Best τb", linestyle = :dash, color = :red)

    # Save the plot as PDF
    savefig(plt, "vinit_vs_taub.pdf")
    display(plt)

    return summary
end


function smooth_VinitSO(df::DataFrame, s::Float64=0.1)
    # Sort by τb to ensure smooth input to spline
    sort!(df, :τb)

    # Extract variables
    τb_vals = df.τb
    VinitSO_vals = df.VinitSO

    # Fit a smooth cubic spline with specified smoothing factor
    spline = Spline1D(τb_vals, VinitSO_vals; k=3, s=s)

    # Save smoothed spline values into a new column
    df.VinitSO_smooth = spline.(τb_vals)

    # Plot the original points and the fitted spline
    τb_fine = range(minimum(τb_vals), stop=maximum(τb_vals), length=200)
    plot(τb_vals, VinitSO_vals, seriestype=:scatter, label="Data", legend=:bottomright)
    plot!(τb_fine, spline.(τb_fine), label="Cubic Spline", lw=2, xlabel="τb", ylabel="VinitSO")

    return df
end


