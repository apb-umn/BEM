

include("OCModelE.jl")
include("FirstOrderApproximation.jl")
using Base.Threads

@info "Using $(Threads.nthreads()) threads"


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
    R, W, Tr = X
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
    @unpack ab_bor_cutoff, ab_col_cutoff, β, σ, α_b, ν, χ, a̲, δ, τb, τc, τw, γ = OCM
    a_ = a_[1]
    a, n, k, yb, nb, c, profit, b, λ, v = x
    λᵉ, vᵉ = yᵉ
    R, W, Tr = X
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
        R*a_ + (1 - τb)*profit + Tr - (1 + τc)*c - (1 + γ)*a,              # (1) Budget constraint
        λ - R*u_c - χ*(mpk - r - δ),                                       # (2) Marginal value of wealth
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
        ret[2] = λ
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
function G(para::OCModel, Ix, A_, X, Xᵉ, Θ)
    @unpack α, δ, τw, τb, τp, τd, τc, b, w, γ, r, g = para
    Ia, In, Ik, Iyb, Inb, Ic, Iprofit, Ib, _, Iv = Ix
    R, W, Tr, Frac_b, V, A, C = X
    TFP = Θ[1]
    A_ = A_[1]
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
    Tb = τb * (Yb - (r + δ) * Kb - w * Nb)

    return [
        Rc - 1 + δ - MPKc,                            # (1) FOC wrt capital
        W - MPNc,                                     # (2) FOC wrt labor
        A - Ia,                                       # (3) Asset market clearing
        C - Ic,                                       # (4) Consumption consistency
        (τc * C + Tp + Td + Tn + Tb) - B*(R - 1 - γ) - Gval - Tr, # (5) Gov budget
        Ib - Frac_b,                                  # (6) Fraction self-employed
        Iv - V                                        # (7) Average utility
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
    inputs.X̄ = getX(OCM)
    inputs.Xlab = [:R, :W, :Tr, :Frac_b, :V, :A, :C]
    inputs.Alab = [:A]
    inputs.Qlab = [:R, :W, :Tr]

    # Distributional objects
    inputs.ω̄, inputs.Λ, inputs.πθ = OCM.ω, OCM.Λ, OCM.πθ
    inputs.Θ̄ = ones(1) * OCM.Θ̄
    inputs.ρ_Θ = ones(1, 1) * 0.8
    inputs.Σ_Θ = ones(1, 1) * 0.017^2

    # Residual functions
    inputs.F = (lθ, a_, c, x, X, yᵉ) -> c == 1 ? Fw(OCM, lθ, a_, x, X, yᵉ) : Fb(OCM, lθ, a_, x, X, yᵉ)
    inputs.G = (Ix, A_, X, Xᵉ, lΘ) -> G(OCM, Ix, A_, X, Xᵉ, lΘ)
    inputs.f = (x⁻, x⁺) -> ff(OCM, x⁻, x⁺)

    return inputs
end

"""
    compute_FO_transition_path(τb_val, τw_val, r_val, tr_val, ω̄_0_base, A_0; T=300)

Compute the first-order transition path of aggregate variables and value functions
in response to a permanent policy change (τb, τw, r, tr), starting from an initial
distribution `ω̄_0_base` and aggregate capital `A_0`.

# Arguments
- `τb_val`: New business tax rate.
- `τw_val`: New wage tax rate.
- `r_val`:  New interest rate (consistent with new steady state).
- `tr_val`: New transfers (consistent with new steady state).
- `ω̄_0_base`: Initial marginal distribution over (a, θ) from old steady state.
- `A_0`: Initial aggregate capital stock.
- `X̄_`: Steady state aggregate variables 
- `T`: Length of transition path (default = 300).

# Returns
- `Xpath`: Transition path of aggregates.
- `Vinit`: Initial average value function under new policy.
"""
function compute_FO_transition_path(τb_val, τw_val, r_val, tr_val, ω̄_0_base, A_0,X̄_; T=300)
    # Initialize model and assign new policy parameters
    OCM = OCModel()
    OCM.τb = τb_val
    OCM.τw = τw_val
    assign!(OCM, r_val, tr_val)

    # Construct approximation objects around new steady state
    inputs = construct_inputs(OCM)
    ZO = ZerothOrderApproximation(inputs)
    computeDerivativesF!(ZO, inputs)
    computeDerivativesG!(ZO, inputs)
    FO = FirstOrderApproximation(ZO, T)

    # Compute linear transition system (some functions redundant by design)
    compute_f_matrices!(FO)
    compute_Lemma3!(FO)
    compute_Lemma4!(FO)
    compute_Corollary2!(FO)
    compute_Proposition1!(FO)
    compute_BB!(FO)

    # Adjust initial distribution using occupational choices implied by new τb
    ω̄ = reshape(OCM.ω, :, 2)
    p̄ = ω̄ ./ sum(ω̄, dims=2)
    p̄[isnan.(p̄[:, 1]), 1] .= 1.0
    p̄[isnan.(p̄[:, 2]), 2] .= 0.0
    ω̄_0 = (p̄ .* ω̄_0_base)[:]  # full initial distribution over (a, θ, occupation)

    # Set initial deviations from new steady state
    FO.X_0 = [A_0] - ZO.P * ZO.X̄
    FO.Θ_0 = [0.0]
    FO.Δ_0 = ω̄_0 - ZO.ω̄

    # Solve forward transition path
    solve_Xt!(FO)

    # Construct final time path and value function at t=0
    Xpath = [X̄_ ZO.X̄ .+ FO.X̂t]
    Vinit = ZO.X̄[inputs.Xlab .== :V] + FO.X̂t[inputs.Xlab .== :V, 1]

    return Xpath, Vinit
end

# === Simulation Setup ===

# --- Old steady state ---
OCM_ = OCModel()
r_val,tr_val = 0.041634407732543365,0.6276923506074257
assign!(OCM_, r_val, tr_val)
inputs_ = construct_inputs(OCM_)

# Extract initial asset distribution and capital
X̄_ = getX(OCM_)
A_0 = X̄_[inputs_.Xlab .== :A][1]                       # Initial capital stock
ω̄_0_base = sum(reshape(OCM_.ω, :, 2), dims=2)         # Distribution over (a, θ)

# --- New steady state ---
τb_val,τw_val = 0.43333333333333335,0.43333333333333335
r_val, tr_val = 0.04223537778372754,0.7674582411891628
# Compute transition path from old to new policy regime
Xpath, Vinit = compute_FO_transition_path(
    τb_val, τw_val, r_val, tr_val, ω̄_0_base, A_0,X̄_;
    T = 300
)

# --- Plots ---


df = DataFrame(Xpath',inputs_.Xlab)
df.t = 0:(size(Xpath,2)-1)

default(linewidth=2)
p1 = plot(df.t, df.A, ylabel="Capital", label="")
p2 = plot(df.t, df.Frac_b, ylabel="Fraction Self Employed", label="")
plot(p1, p2, layout=(2, 1), size=(800, 600), legend=:topright)



default(linewidth=2)
p1 = plot(df.t, df.R, ylabel="R", label="")
p2 = plot(df.t, df.Tr, ylabel="Tr", label="")
plot(p1, p2, layout=(2, 1), size=(800, 600), legend=:topright)
# --- Save results ---
CSV.write("df_transition.csv", df)


    # new ss

# results_df = CSV.read("grid_results.csv", DataFrame)
# results_df[!,:value]=ones(length(results_df.r))*NaN
# using Base.Threads

# n = nrow(results_df)
# values = Vector{Union{Float64, Missing}}(undef, n)

# Threads.@threads for i in 1:n
#     row = results_df[i, :]
#     if !ismissing(row.r) && !ismissing(row.tr)
#         try
#             println("Thread $(threadid()) processing row $i")
#             Xpath, Vinit = compute_FO_transition_path(
#                 row.τb, row.τw, row.r, row.tr, ω̄_0_base, A_0; T=300)
#             values[i] = Vinit[1]
#         catch e
#             @warn "Error in row $i on thread $(threadid()): $e"
#             values[i] = missing
#         end
#     else
#         values[i] = missing
#     end
# end

# # Assign the computed values to a new column
# results_df.value = values
# CSV.write("grid_results_with_values.csv", results_df)


# save CSV
#CSV.write("df_transition.csv", df)
# 

