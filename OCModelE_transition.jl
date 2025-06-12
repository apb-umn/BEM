

include("OCModelE.jl")
include("FirstOrderApproximation.jl")
 

"""
    Fw(OCM::OCModel, lθ, a_, x, X, yᵉ)

Equilibrium conditions for a worker. This function returns residuals for the system of
equations characterizing a worker's optimization problem, including consumption,
savings, labor, and value function conditions.

"""
function Fw(OCM::OCModel, lθ, a_, x, X, yᵉ)
    @unpack aw_bor_cutoff, β, σ, a̲, τc, τw, γ = OCM
    a_ = a_[1]
    a, n, k, yb, nb, c, profit, b, λ, v = x
    λᵉ, vᵉ = yᵉ
    R, W, Tr,Taub = X
    _, ϵ = exp.(lθ)

    # Marginal utility and utility
    u_c = c^(-σ)
    u = c^(1 - σ) / (1 - σ)

    # FOC system
    ret = [
        R*a_ + (1 - τw)*W*ϵ + Tr - (1 + τc)*c - (1 + γ)*a,    # (1) Budget constraint
        λ - R*u_c,                                            # (2) Envelope condition: ∂V/∂a
        β*λᵉ - u_c,                                           # (3) Euler equation
        u + β*vᵉ - v,                                         # (4) Bellman equation
        n + ϵ,                                                # (5) Labor supply condition (ϵ = disutility shock)
        k, yb, nb, profit, b                                  # (6–10) unused in worker case
    ]

    # Replace Euler with borrowing constraint if a_ below cutoff
    if a_ <= aw_bor_cutoff[lθ]
        ret[3] = a̲ - a  # (3b) Binding borrowing constraint
    end

    return ret
end




"""
    Fb(OCM::OCModel, lθ, a_, x, X, yᵉ)

Equilibrium conditions for a business owner. Includes budget constraint,
first-order conditions for inputs, and collateral/borrowing constraints.
"""
function Fb(OCM::OCModel, lθ, a_, x, X, yᵉ)
    @unpack ab_bor_cutoff, ab_col_cutoff, τb,β, σ, α_b, ν, χ, a̲, δ , τc, τw, γ = OCM
    a_ = a_[1]
    a, n, k, yb, nb, c, profit, b, λ, v = x
    λᵉ, vᵉ = yᵉ
    R, W, Tr,Taub = X
    r = R - 1
    z, _ = exp.(lθ)



    # Marginal products
    mpk = α_b * z * k^(α_b - 1) * n^ν       # MPK: ∂Y/∂K
    mpn = ν * z * k^α_b * n^(ν - 1)         # MPN: ∂Y/∂N
 

    # Utility
    u_c = c^(-σ)
    u = c^(1 - σ) / (1 - σ)

    # System of residuals
    ret = [
        R*a_ + (1 - Taub)*profit + Tr - (1 + τc)*c - (1 + γ)*a,            # (1) Budget constraint
        λ - R*u_c - u_c*χ*(mpk - r - δ)*(1-Taub),                          # (2) Marginal value of wealth
        β*λᵉ - u_c,                                                        # (3) Euler equation
        u + β*vᵉ - v,                                                      # (4) Value function
        mpn - W,                                                           # (5) Labor FOC
        nb - n,                                                            # (6) Consistency: hired labor = choice
        mpk - r - δ,                                                       # (7) Capital FOC
        yb - z*k^α_b*n^ν,                                                  # (8) Output production function
        profit - (yb - δ*k - r*k - W*n),                                   # (9) Profit definition
        b - 1                                                              # (10) Occupational identity
    ]

    if a_ <= ab_bor_cutoff[lθ]
        ret[3] = a̲ - a
        ret[7] = k - χ*a_
    elseif (a_ <= ab_col_cutoff[lθ]) && (a_ > 0)
        ret[7] = k - χ*a_
    end

    if a_ == 0
        ret[2] = λ- R*u_c - u_c*χ*( α_b * z * (max(k,1e-4))^(α_b - 1) * (max(n,1e-4))^ν - r - δ)  # Marginal value of wealth at zero assets
        ret[5] = n
        ret[6] = nb
        ret[7] = k
        ret[8] = yb
        ret[9] = profit
    end

    return ret
end


"""
    G(para::OCModel, Ix, A_, X, Xᵉ, Θ)

Aggregate equilibrium conditions. Ensures market clearing, government budget
balance, and correct pricing in production sectors.
"""
function G(para::OCModel, Ix, X_, X, Xᵉ, Θ)
    @unpack α, δ, τw, τp, τd, τc, b, w, γ, r, g, τb, ρ_τ = para
    Ia, In, Ik, Iyb, Inb, Ic, Iprofit, Ib, _, Iv = Ix
    R, W, Tr, Frac_b, V, A, C,Taub = X
    TFP = Θ[1]
    A_ = X_[1]
    Taub_=X_[2]
    B = b
    Gval = g

    # Business sector aggregates
    Yb = Iyb
    Kb = Ik
    Nb = Inb

    # Corporate sector
    Nc = .-In
    Kc = (A_ - Ik - B) / (1 - τd)
    MPKc = α * TFP * Kc^(α - 1) * Nc^(1 - α)
    MPNc = (1 - α) * TFP * Kc^α * Nc^(-α)
    Yc = TFP * Kc^α * Nc^(1 - α)
    Rc = (R - 1) / (1 - τp) + 1

    # Government budget components
    Tp = τp * (Yc - w * Nc - δ * Kc)
    Td = τd * (Yc - w * Nc - (γ + δ) * Kc - Tp)
    Tn = τw * w * (Nc + Nb)
    Tb = Taub * (Yb - (r + δ) * Kb - w * Nb)

    return [
        Rc - 1 + δ - MPKc,                            # (1) FOC wrt capital
        W - MPNc,                                     # (2) FOC wrt labor
        A - Ia,                                       # (3) Asset market clearing
        C - Ic,                                       # (4) Consumption consistency
        (τc * C + Tp + Td + Tn + Tb) - B*(R - 1 - γ) - Gval - Tr, # (5) Gov budget
        Ib - Frac_b,                                  # (6) Fraction self-employed
        Iv - V,                                       # (7) Average utility
        Taub-(τb+ρ_τ*(Taub_-τb)),                     # (8) Business tax rate consistency
    ]
end



"""
    ff(para::OCModel, x⁻, x⁺)

Smooth discrete occupational choice via log-sum (softmax).
Returns expected marginal utility and expected value.
"""
function ff(para::OCModel, x⁻, x⁺)
    @unpack σ_ε = para
    _,_,_,_,_,_,_,_,λ⁻,v⁻ = x⁻
    _,_,_,_,_,_,_,_,λ⁺,v⁺ = x⁺

    p = 1 / (1 + exp((v⁺ - v⁻)/σ_ε))

    if p < 1e-9
        return [λ⁺, v⁺]
    elseif p > 1 - 1e-9
        return [λ⁻, v⁻]
    else
        Ev = v⁻ + σ_ε * log(1 + exp((v⁺ - v⁻)/σ_ε))
        return [λ⁻*p + λ⁺*(1-p), Ev]
    end
end


"""
    construct_inputs(OCM)

Create and return `Inputs` object with model functions, grids, and equilibrium mappings.
Used to compute steady state and approximations.
"""
function construct_inputs(OCM)
    inputs = Inputs()

    # Policy functions and grids
    inputs.xf = get_policy_functions(OCM)
    inputs.aknots, inputs.ka, inputs.aθc_sp, inputs.aθc_Ω, inputs.ℵ = get_grids(OCM)
    inputs.xlab = [:a, :n, :k, :yb, :nb, :c, :profit, :b, :λ, :v]
    inputs.alab = [:a]
    inputs.κlab = [:v]
    inputs.yᵉlab = [:λ, :v]

    # Choice probabilities (Gumbel CDF)
    inputs.Γf = κ -> 1 / (1 + exp(κ / OCM.σ_ε))
    inputs.dΓf = κ -> -(1 / OCM.σ_ε) * exp(κ / OCM.σ_ε) / (1 + exp(κ / OCM.σ_ε))^2

    # Aggregates and equilibrium labels
    inputs.X̄ = [getX(OCM);OCM.τb]
    inputs.Xlab = [:R, :W, :Tr, :Frac_b, :V, :A, :C,:Taub]
    inputs.Alab = [:A,:Taub]
    inputs.Qlab = [:R, :W, :Tr,:Taub]

    # Distributional objects
    inputs.ω̄, inputs.Λ, inputs.πθ = OCM.ω, OCM.Λ, OCM.πθ
    inputs.Θ̄ = ones(1) * OCM.Θ̄
    inputs.ρ_Θ = ones(1, 1) * 0.8
    inputs.Σ_Θ = ones(1, 1) * 0.017^2

    # Residual functions
    inputs.F = (lθ, a_, c, x, X, yᵉ) -> c == 1 ? Fw(OCM, lθ, a_, x, X, yᵉ) : Fb(OCM, lθ, a_, x, X, yᵉ)
    inputs.G = (Ix, X_, X, Xᵉ, lΘ) -> G(OCM, Ix, X_, X, Xᵉ, lΘ)
    inputs.f = (x⁻, x⁺) -> ff(OCM, x⁻, x⁺)

    return inputs
end


using Plots, DataFrames

function plot_transition_comparison_dfs(df_slow::DataFrame, df_fast::DataFrame; savepath::String="transition_comparison.pdf")
    # Add time column if not present
    if :t ∉ names(df_slow)
        df_slow.t = 0:(nrow(df_slow)-1)
    end
    if :t ∉ names(df_fast)
        df_fast.t = 0:(nrow(df_fast)-1)
    end

    # Variables to plot and their titles
    variables = [:A, :C, :Frac_b, :Tr, :W, :Taub]
    titles = [
        "Capital (A)", "Consumption (C)", "Fraction Borrowers (Frac_b)",
        "Transfers (Tr)", "Wage (W)", "Capital Tax (Taub)"
    ]

    # Create the plot layout
    plt = plot(layout = (3, 2), size=(1000, 800))

    # Plot each variable
    for (i, var) in enumerate(variables)
        plot!(plt[i], df_slow.t, df_slow[!, var], label = "Slow Reform", linestyle=:dash, lw=2)
        plot!(plt[i], df_fast.t, df_fast[!, var], label = "Fast Reform", linestyle=:solid, lw=2)
        plot!(plt[i], xlabel="Time", ylabel=string(var), title=titles[i])
    end

    # Save and display
    savefig(plt, savepath)
    display(plt)
end



function analyze_optimal_taub(file::String)
    # Step 1: Read the data
    df = CSV.read(file, DataFrame)

    # Step 2: Find the best τb using Vinit and Vss
    idx_vinit = argmax(df.Vinit)
    idx_vss   = argmax(df.Vss)

    best_taub_vinit = df.τb[idx_vinit]
    best_taub_vss   = df.τb[idx_vss]

    # Step 3: Construct summary table
    summary = DataFrame(
        Criterion = ["Best by Vss+Transition", "Best by Vss"],
        τb = [best_taub_vinit, best_taub_vss],
        Vinit = [df.Vinit[idx_vinit], df.Vinit[idx_vss]],
        Vss = [df.Vss[idx_vinit], df.Vss[idx_vss]]
    )

    println("\nSummary Table:")
    pretty_table(summary; formatters = ft_printf("%.4f"))

    # Save summary table to LaTeX file
    open("optimal_tau_summary.tex", "w") do io
        pretty_table(
            io, summary;
            header = names(summary),
            backend = Val(:latex),
            tf = tf_latex_booktabs,   # <- FIXED: no parentheses
            formatters = ft_printf("%.4f")
        )
    end

    # Step 4: Plot Vinit vs τb with optima marked
    default(linewidth=2)
    plt = plot(df.τb, df.Vinit, label = "V₀", xlabel = "τb", ylabel = "V",
               title = "Value vs τb", legend = :bottomright, grid = true)

    vline!([best_taub_vinit], label = "Best τb", linestyle = :dash, color = :red)

    # Save the plot as PDF
    savefig(plt, "vinit_vs_taub.pdf")
    display(plt)

    return summary
end
