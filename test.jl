
include("OCModelEGMInputs.jl")
include("OCModelEGM.jl")
include("OCModelEGM_transition.jl") # has the F,G,ff and several helper functions

# Define parameters
τb_val = 0.40
ρ_τ_val_fast = 0.0
ρ_τ_val_slow = 0.9

# Create output filenames using interpolation
filenamefast = "df_transition_fast_$(round(τb_val, digits=2)).csv"
filenameslow = "df_transition_slow_$(round(τb_val, digits=2)).csv"
saveplotfilename = "transition_comparison_$(round(τb_val, digits=2)).pdf"

println("Setting up old steady state (takes a few minutes)...")
OCM_old = OCModel()
setup!(OCM_old)
OCM_old.r = 0.03867770200367673
OCM_old.tr = 0.46806124541903205
    _, X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old = setup_old_steady_state!(OCM_old)
println("Old steady state setup complete.")

println("Setting up new steady state with τb = $τb_val (takes a few minutes)...")
OCM_new, _ = setup_new_steady_state(τb_val, OCM_old.τw, OCM_old)
println("New steady state setup complete.")

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

println("→ Setting up first-order approximation object...")
FO = FirstOrderApproximation(ZO, OCM_new.T)
println("...done")

println("→ Computing x,M,L,Js components ( bulk of the calclations )...")
compute_f_matrices!(FO)
compute_Lemma3!(FO)
compute_Lemma4!(FO)
compute_Corollary2!(FO)
compute_Proposition1!(FO)
compute_BB!(FO)
println("...done")