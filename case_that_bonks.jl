
### parameters that bonk


ρ_τ_vals=[0.0]

ρ_τ_val=0.0
τb_val=0.308 # compute_BB!(FO) produces singularity  exception
#τb_val=0.4060 # diverges to inf
#τb_val=0.43 # seems ok

include("OCModelEGM_driver.jl")
include("OCModelEGM_transition.jl") # has the F,G,ff and several helper functions
 include("OCModelEGMInputs.jl")
#include("OCModelEGM_opttaxmpi.jl")
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
    function setup_new_steady_state(τb, τw, ρ_τ, OCM_old)
        OCM = deepcopy(OCM_old)
        OCM.τb = τb
        OCM.τw = τw
        OCM.ρ_τ = ρ_τ
        OCM.iprint = 1
        OCM.ibise = 0
        ss, lev, shr, res = solvess!(OCM)
        updatecutoffs!(OCM)
        Xss = [getX(OCM); OCM.τb]
        return OCM, Xss
    end


println("Setting up old steady state (takes a few minutes) on master node...")
OCM_old = OCModel()
OCM_old.σ_ε = 0.01
OCM_old.Θ̄ = 0.655
setup!(OCM_old)
OCM_old.r = 0.038827378860131295
OCM_old.tr = 0.5490743758858707
inputs_old, X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old = setup_old_steady_state!(OCM_old)
println("Old steady state setup complete..")

state = SharedState(OCM_old, inputs_old, X̄_old, A_old, Taub_old, ω̄_0_old, ρ_τ_vals)




OCM_old   = state.OCM_old
inputs    = state.inputs_old
X̄         = state.X̄_old
A         = state.A_old
Taub      = state.Taub_old
ω̄        = state.ω̄_old
ρ_τ_vals  = state.ρ_τ_vals
# --- Baseline ergodic objects (CRRA, σ ≠ 1) ---
cdst, _, _, _, _, _, _ = dist!(OCM_old)             # steady-state consumption
σ  = OCM_old.σ
β  = OCM_old.βV
ω  = OCM_old.ω
ω  = ω ./ sum(ω)                                    # ensure weights sum to 1
Vss_old = X̄[inputs.Xlab .== :V][1]
udst = cdst.^(1 - σ) ./ (1 - σ)                     # per-period utility at each state
Ubar = dot(ω, udst)                                 # ergodic average per-period utility

Wbar   = Ubar / (1 - β)                             # = \bar U / (1-β)
Resbar = Vss_old - Wbar                             # = \overline{Residual}/(1-β)


OCM_new, Xss = setup_new_steady_state(τb_val, OCM_old.τw, ρ_τ_val, OCM_old)
Vss = Xss[inputs.Xlab .== :V][1]

X̄_0, A_0, Taub_0, ω̄_0_base, OCM_new = X̄, A, Taub, ω̄, OCM_new

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


df = DataFrame(XpathSO', inputs.Xlab)
df.t = 0:(size(XpathSO, 2) - 1)

    # Variables to plot and their titles
    variables = [:A, :C, :Frac_b, :Tr, :W, :Taub]
    titles = [
        "Capital (A)", "Consumption (C)", "Fraction Borrowers (Frac_b)",
        "Transfers (Tr)", "Wage (W)", "Business Tax (Taub)"
    ]

    # Create the plot layout
    plt = plot(layout = (3, 2), size=(1000, 800))

    # Plot each variable
    for (i, var) in enumerate(variables)
        plot!(plt[i], df.t, df[!, var], linestyle=:solid, lw=2)
    end
        display(plt)


        # scracth work area for testing bonking parameters


        using Plots

"""
Side-by-side occupation panels per shock level:
  For each selected shock s, show λ vs a for c=1 (worker) and c=2 (business).

Requirements on OCM:
  - OCM.wf.λ, OCM.bf.λ :: Vector{<:Function} (or splines) indexed by rows of OCM.lθ
  - OCM.lθ             :: Matrix (nθrows × ≥2), e.g. columns [θ_level, θ_shock]

Arguments:
  OCM, aθc_sp  : long-form table where each row j aligns with policy columns
  acol         : column index of a in aθc_sp (default 1)
  θ1col        : column index of θ_level in aθc_sp (default 2)
  shockcol     : column index of θ_shock in aθc_sp (default 3)

Keywords:
  nshocks::Int = 5           number of shock levels (evenly spaced across unique values)
  a_min::Real=-Inf, a_max::Real=Inf   x-axis window
"""
function plot_lambda_side_by_side_per_shock(OCM, aθc_sp;
        acol::Int=1, θ1col::Int=2, shockcol::Int=3,
        nshocks::Int=5, a_min::Real=-Inf, a_max::Real=Inf)

    wf, bf = OCM.wf, OCM.bf
    θgrid  = OCM.lθ
    @assert size(θgrid,2) ≥ 2 "OCM.lθ must have ≥2 columns (θ_level, θ_shock)."

    a  = @view aθc_sp[:, acol]
    θ1 = @view aθc_sp[:, θ1col]
    s  = @view aθc_sp[:, shockcol]

    # Pick representative shocks
    shock_levels = sort(unique(s))
    n_pick = min(nshocks, length(shock_levels))
    s_choices = shock_levels[round.(Int, range(1, length(shock_levels); length=n_pick))]
    println("Shock panels: ", s_choices)

    # Nearest θ-row on first two dims (θ_level, θ_shock)
    @inline function nearest_θ_index(θ1val::Real, sval::Real)
        dmin = Inf; imin = 0
        @inbounds for i in 1:size(θgrid,1)
            d = (θgrid[i,1]-θ1val)^2 + (θgrid[i,2]-sval)^2
            if d < dmin; dmin = d; imin = i; end
        end
        return imin
    end

    panels = Any[]
    for sₜ in s_choices
        js = findall(==(sₜ), s)
        if isempty(js)
            @warn "No grid points for shock=$sₜ; skipping."
            push!(panels, plot(title="No data for shock=$sₜ")); push!(panels, plot()) # keep layout shape
            continue
        end
        js = js[sortperm(a[js])]
        avec = a[js]

        # Build λ series for both occupations
        λ1 = similar(avec); λ2 = similar(avec)
        @inbounds for (k, j) in enumerate(js)
            iθ = nearest_θ_index(θ1[j], s[j])
            λ1[k] = wf.λ[iθ](a[j])   # worker c=1
            λ2[k] = bf.λ[iθ](a[j])   # business c=2
        end

        # Filters (window + finite)
        inwin = (avec .>= a_min) .& (avec .<= a_max)
        good1 = inwin .& isfinite.(avec) .& isfinite.(λ1)
        good2 = inwin .& isfinite.(avec) .& isfinite.(λ2)

        p_left = plot(title="shock=$(round(sₜ, digits=6)) • c=1 (worker)",
                      xlabel="a", ylabel="λ", legend=false)
        p_right = plot(title="shock=$(round(sₜ, digits=6)) • c=2 (business)",
                       xlabel="a", ylabel="λ", legend=false)
        any(good1) && plot!(p_left,  avec[good1], λ1[good1]; lw=2)
        any(good2) && plot!(p_right, avec[good2], λ2[good2]; lw=2)

        if isfinite(a_min) || isfinite(a_max)
            xlims!(p_left,  (isfinite(a_min) ? a_min : minimum(avec), isfinite(a_max) ? a_max : maximum(avec)))
            xlims!(p_right, (isfinite(a_min) ? a_min : minimum(avec), isfinite(a_max) ? a_max : maximum(avec)))
        end

        push!(panels, p_left, p_right)
    end

    # Arrange as (#shocks rows) × 2 columns (c=1 | c=2), share x
    combo = plot(panels...; layout=(length(s_choices), 2), size=(1000, 260*length(s_choices)), link=:x)
    display(combo)
    return combo
end


# Five shock levels, full range
#plot_lambda_side_by_side_per_shock(OCM, aθc_sp)

# Three shocks, zoom to a≤50
plot_lambda_from_OCM(OCM; nshocks=5, a_max=5) # zoom on a ≤ 50

