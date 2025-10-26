
### parameters that bonk


ρ_τ_vals=[0.0]

ρ_τ_val=0.0
#τb_val=0.308 # compute_BB!(FO) produces singularity  exception
#τb_val=0.32 # checking

τb_val=0.4060 # diverges to inf
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
OCM_old.Na = 150
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



# Five shock levels, full range
#plot_lambda_side_by_side_per_shock(OCM, aθc_sp)

# Three shocks, zoom to a≤50
plot_lambda_from_OCM(OCM_new; nshocks=5, a_max=1) # zoom on a ≤ 50

