# Modular transition simulation script

include("OCModelE_transition.jl")
using CSV, DataFrames, Plots
using Base.Threads
println("Number of threads available: ", Threads.nthreads())


function setup_old_steady_state()
    OCM = OCModel()
    setup!(OCM)
    OCM.r = 0.041760472228065636 #baseline
    OCM.tr = 0.652573199821719 #baseline
    OCM.ibise = 0
    OCM.iprint = 0
    solvess!(OCM)
    assign!(OCM, OCM.r, OCM.tr)
    inputs = construct_inputs(OCM)
    X̄_0 = [getX(OCM); OCM.τb]
    A_0 = X̄_0[inputs.Xlab .== :A][1]
    Taub_0 = X̄_0[inputs.Xlab .== :Taub][1]
    ω̄_0_base = sum(reshape(OCM.ω, :, 2), dims=2)
    inputs_0 = construct_inputs(OCM)
    ZO_0 = ZerothOrderApproximation(inputs_0)
    Ix̄_0 = ZO_0.x̄*ZO_0.Φ*ZO_0.ω̄
    return OCM, inputs, X̄_0, Ix̄_0,A_0, Taub_0, ω̄_0_base
end

function setup_new_steady_state(τb, τw, OCM_old)
    OCM = deepcopy(OCM_old)
    OCM.τb = τb
    OCM.τw = τw
    OCM.iprint = 1
    OCM.ibise = 1
    assign!(OCM, OCM_old.r, OCM_old.tr)
    ss, lev, shr, res = solvess!(OCM)
    Xss = [getX(OCM); OCM.τb]
    return OCM, Xss
end

function perform_transition_analysis(X̄_0,Ix̄_0,A_0, Taub_0, ω̄_0_base, OCM_new)
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
    println("...done")


    println("→ Constructing Ix paths...")
    compute_x̂t_ω̂t!(FO)
    IX̂=compute_Ixt(FO)
    Ix̄ = ZO.x̄*ZO.Φ*ZO.ω̄
    Ixpath=[Ix̄_0 Ix̄.+IX̂]
    println("...done ✅")

    return Xpath, Ixpath,inputs, Vinit
end



function save_stuff(Xpath, Ixpath, inputs, Vss, Vinit)
    println("SS value: $(Vss)")
    println("SS + transition value: $(Vinit[1])")

    df = DataFrame(Xpath', inputs.Xlab)
    df.t = 0:(size(Xpath, 2) - 1)

    df=hcat(df, DataFrame(Ixpath', map(x -> Symbol("I", String(x)), inputs.xlab)))

    default(linewidth = 2)
    p1 = plot(df.t, df.A, ylabel = "Capital", label = "")
    p2 = plot(df.t, df.Frac_b, ylabel = "Fraction Self Employed", label = "")
    display(plot(p1, p2, layout = (2, 1), size = (800, 600), legend = :topright))

    p3 = plot(df.t, df.W, ylabel = "Wage", label = "")
    display(p3)

    p4 = plot(df.t, df.C, ylabel = "Consumption", label = "")
    display(p4)

    p5 = plot(df.t, df.R, ylabel = "R", label = "")
    p6 = plot(df.t, df.Tr, ylabel = "Tr", label = "")
    display(plot(p5, p6, layout = (2, 1), size = (800, 600), legend = :topright))

    return df
end

# === Run all ===
τb_val = 0.634783
ρ_τ_val_fast = 0.0
ρ_τ_val_slow = 0.95
filenamefast = "df_transition_fast_opt.csv"
filenameslow = "df_transition_slow_opt.csv"
saveplotfilename = "transition_comparison_opt.pdf"



println("Setting up old steady state...")

OCM_old, inputs_old, X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old = setup_old_steady_state()
println("Old steady state setup complete.")


println("Setting up new steady state with τb = $τb_val...")
OCM_new, Xss = setup_new_steady_state(τb_val, OCM_old.τw, OCM_old)
println("New steady state setup complete.")

compare_moments(OCM_old, OCM_new)

OCM_new.ρ_τ = ρ_τ_val_fast
println("Performing transition analysis with ρ_τ = $ρ_τ_val_fast...")
Xpath, Ixpath, inputs, Vinit = perform_transition_analysis(X̄_old, Ix̄_old,A_old, Taub_old, ω̄_0_old, OCM_new)
println("Transition analysis complete.")

println("Transition analysis results:")
Vss = Xss[inputs.Xlab.==:V] 
println("SS value: $(Vss)")
println("SS + transition value for rho_t = $(ρ_τ_val_fast) is $(Vinit[1])")
df_transition_fast=save_stuff(Xpath, Ixpath, inputs, Vss, Vinit)
CSV.write(filenamefast, df_transition_fast)
println("Results saved to $(filenamefast)")

OCM_new.ρ_τ = ρ_τ_val_slow
println("Performing transition analysis with ρ_τ = $ρ_τ_val_slow...")

Xpath, Ixpath, inputs, Vinit = perform_transition_analysis(X̄_old, Ix̄_old,A_old, Taub_old, ω̄_0_old, OCM_new)
println("Transition analysis complete.")

println("Transition analysis results:")
Vss = Xss[inputs.Xlab.==:V] 
println("SS value: $(Vss)")
println("SS + transition value for rho_t = $(ρ_τ_val_slow) is $(Vinit[1])")
df_transition_slow=save_stuff(Xpath, Ixpath, inputs, Vss, Vinit)

CSV.write(filenameslow, df_transition_slow)
println("Results saved to $(filenameslow)")

# === End of run ===
println("Transition analysis completed successfully.")
println("All results saved to CSV files.")

# === Plotting ===
plot_transition_comparison_dfs(df_transition_fast,df_transition_slow, savepath=saveplotfilename)
println("Transition comparison plot saved to $(saveplotfilename)")
# === End of script ===
