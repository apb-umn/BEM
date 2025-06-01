include("OCModelE.jl")              # Load model specification
include("FirstOrderApproximation.jl") # Load approximation routines
using Base.Threads                  # Enable multi-threading

@info "Using $(Threads.nthreads()) threads"

# === Worker Optimization ===
"""
    Fw(OCM::OCModel, lθ, a_, x, X, yᵉ)

Equilibrium conditions for a worker. Returns residuals for budget, Euler, envelope, and Bellman equations.
"""
function Fw(OCM::OCModel, lθ, a_, x, X, yᵉ)
    @unpack aw_bor_cutoff, β, σ, a̲, τc, τw, γ, α, Θ̄, τp, δ = OCM
    a_ = a_[1]
    a, n, k, yb, nb, c, profit, b, λ, v = x
    λᵉ, vᵉ = yᵉ
    R, Tr = X
    _, ϵ = exp.(lθ)
    r = R - 1
    W = (1 - α) * Θ̄ * ((r / (1 - τp) + δ) / (Θ̄ * α))^(α / (α - 1))
    u_c = c^(-σ)
    u = c^(1 - σ) / (1 - σ)
    ret = [
        R*a_ + (1 - τw)*W*ϵ + Tr - (1 + τc)*c - (1 + γ)*a,  # (1) Budget constraint
        λ - R*u_c,                                          # (2) Envelope
        β*λᵉ - u_c,                                         # (3) Euler
        u + β*vᵉ - v,                                      # (4) Bellman
        n + ϵ,                                             # (5) Labor supply
        k, yb, nb, profit, b                               # (6–10) Unused
    ]
    if a_ <= aw_bor_cutoff[lθ]
        ret[3] = a̲ - a  # (3b) Borrowing constraint binds
    end
    return ret
end

# === Business Owner Optimization ===
"""
    Fb(OCM::OCModel, lθ, a_, x, X, yᵉ)

Equilibrium conditions for a business owner. Includes borrowing and collateral constraints.
"""
function Fb(OCM::OCModel, lθ, a_, x, X, yᵉ)
    @unpack ab_bor_cutoff, ab_col_cutoff, β, σ, α_b, ν, χ, a̲, τb, τc, τw, γ, Θ̄, τp, δ, α = OCM
    a_ = a_[1]
    a, n, k, yb, nb, c, profit, b, λ, v = x
    λᵉ, vᵉ = yᵉ
    R, Tr = X
    r = R - 1
    z, _ = exp.(lθ)
    W = (1 - α) * Θ̄ * ((r / (1 - τp) + δ) / (Θ̄ * α))^(α / (α - 1))
    mpk = α_b * z * k^(α_b - 1) * n^ν
    mpn = ν * z * k^α_b * n^(ν - 1)
    u_c = c^(-σ)
    u = c^(1 - σ) / (1 - σ)
    ret = [
        R*a_ + (1 - τb)*profit + Tr - (1 + τc)*c - (1 + γ)*a,    # (1)
        λ - R*u_c - u_c*χ*(mpk - r - δ),                         # (2)
        β*λᵉ - u_c,                                              # (3)
        u + β*vᵉ - v,                                           # (4)
        mpn - W,                                                # (5)
        nb - n,                                                 # (6)
        mpk - r - δ,                                            # (7)
        yb - z*k^α_b*n^ν,                                       # (8)
        profit - (z*k^α_b*n^ν - δ*k - r*k - W*n),                # (9)
        b - 1                                                   # (10)
    ]
    if a_ <= ab_bor_cutoff[lθ]
        ret[3] = a̲ - a
        ret[7] = k - χ*a_
    elseif (a_ > ab_bor_cutoff[lθ]) && (a_ <= ab_col_cutoff[lθ]) && (a_ > 0)
        ret[7] = k - χ*a_
    end
    if a_ == 0
        ret[2] = λ- R*u_c - u_c*χ*( α_b * z * (max(k,1e-4))^(α_b - 1) * (max(n,1e-4))^ν - r - δ)  # Marginal value of wealth at zero assets
        ret[5:9] .= [n, nb, k, yb, profit]
    end
    return ret
end
# === Aggregate Equilibrium Conditions ===
"""
    G(para::OCModel, Ix, A_, X, Xᵉ, Θ)

Aggregate equilibrium conditions. Ensures:
1. Asset market clearing,
2. Government budget balance,
3. Consumption consistency.

# Arguments
- `para`: Model parameters (`OCModel` type).
- `Ix`: Vector of integrals over individual decisions (e.g., Ia, In, Ik, etc.).
- `A_`: Lagged aggregate asset stock.
- `X`: Current aggregate state (R, Tr, A).
- `Xᵉ`: Expected future aggregates (not used here).
- `Θ`: Vector of aggregate shocks (e.g., TFP).

# Returns
- Vector of residuals for the 3 general equilibrium conditions.
"""
function G(para::OCModel, Ix, A_, X, Xᵉ, Θ)
    @unpack α, δ, τw, τb, τp, τd, τc, b, w, γ, r, g = para

    # === Unpack aggregate integrals ===
    Ia, In, Ik, Iyb, Inb, Ic, Iprofit, Ib, _, Iv = Ix
    R, Tr, A = X
    TFP = Θ[1]               # Current TFP
    A_ = A_[1]               # Lagged aggregate assets
    B = b                    # Government debt
    r = R - 1.0              # Interest rate on capital
    Rc = (R - 1) / (1 - τp) + 1

    # === Labor and capital in corporate and business sectors ===
    Nc = -In                 # Corporate labor (In is negative by convention)
    Nb = Inb                 # Business labor
    Kb = Ik                  # Business capital
    Kc = (A_ - Ik - B) / (1 - τd)   #  corporate capital (from assets)
    MPKc = α * TFP * Kc^(α - 1) * Nc^(1 - α)

    # === Output in each sector ===
    Yc = TFP * Kc^α * Nc^(1 - α)    # Corporate output
    Yb = Iyb                         # Business output (from integral)


    # === Government tax revenues ===
    Tp = τp * (Yc - w * Nc - δ * Kc)                    # Capital income tax
    Td = τd * (Yc - w * Nc - (γ + δ) * Kc - Tp)         # Dividend tax
    Tn = τw * w * (Nc + Nb)                             # Labor income tax
    Tb = τb * (Yb - (r + δ) * Kb - w * Nb)              # Business profit tax

    # === General equilibrium residuals ===
    return [
        A - Ia,  # (1) Asset market clearing: actual assets vs. distribution-integrated assets
        τc * Ic + Tp + Td + Tn + Tb - B * (R - 1 - γ) - g - Tr,  # (2) Government budget constraint
        Rc - 1 + δ - MPKc,                            # (3) FOC wrt capital
    ]
end


# === Occupational Choice Function ===
"""
    ff(para::OCModel, x⁻, x⁺)

Log-sum discrete choice returns expected marginal utility and value.
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
        return [λ⁻ * p + λ⁺ * (1 - p), Ev]
    end
end

# === Input Construction ===
"""
    construct_inputs(OCM)

Builds and returns the Inputs object with residuals, grids, and labels.
"""
function construct_inputs(OCM)
    inputs = Inputs()
    inputs.xf = get_policy_functions(OCM)
    inputs.aknots, inputs.ka, inputs.aθc_sp, inputs.aθc_Ω, inputs.ℵ = get_grids(OCM)
    inputs.xlab = [:a, :n, :k, :yb, :nb, :c, :profit, :b, :λ, :v]
    inputs.alab = [:a]
    inputs.κlab = [:v]
    inputs.yᵉlab = [:λ, :v]
    inputs.Γf = κ -> 1 / (1 + exp(κ / OCM.σ_ε))
    inputs.dΓf = κ -> -(1 / OCM.σ_ε) * exp(κ / OCM.σ_ε) / (1 + exp(κ / OCM.σ_ε))^2
    inputs.X̄ = getX(OCM)[[1, 3, 6]]
    inputs.Xlab = [:R, :Tr, :A]
    inputs.Alab = [:A]
    inputs.Qlab = [:R, :Tr]
    inputs.ω̄, inputs.Λ, inputs.πθ = OCM.ω, OCM.Λ, OCM.πθ
    inputs.Θ̄ = ones(1) * OCM.Θ̄
    inputs.ρ_Θ = ones(1, 1) * 0.8
    inputs.Σ_Θ = ones(1, 1) * 0.017^2
    inputs.F = (lθ, a_, c, x, X, yᵉ) -> c == 1 ? Fw(OCM, lθ, a_, x, X, yᵉ) : Fb(OCM, lθ, a_, x, X, yᵉ)
    inputs.G = (Ix, A_, X, Xᵉ, lΘ) -> G(OCM, Ix, A_, X, Xᵉ, lΘ)
    inputs.f = (x⁻, x⁺) -> ff(OCM, x⁻, x⁺)
    return inputs
end

# === Transition Path Solver ===
"""
    compute_FO_transition_path(τb_val, τw_val, r_val, tr_val, ω̄_0_base, A_0; T=300)

Compute transition path from initial capital and distribution under new tax policy.
"""
function compute_FO_transition_path(τb_val, τw_val, r_val, tr_val, ω̄_0_base, A_0,X̄_; T=300)
    OCM = OCModel()
    OCM.τb = τb_val
    OCM.τw = τw_val
    assign!(OCM, r_val, tr_val)
    inputs = construct_inputs(OCM)
    ZO = ZerothOrderApproximation(inputs)
    computeDerivativesF!(ZO, inputs)
    computeDerivativesG!(ZO, inputs)
    FO = FirstOrderApproximation(ZO, T)
    compute_f_matrices!(FO)
    compute_Lemma3!(FO)
    compute_Lemma4!(FO)
    compute_Corollary2!(FO)
    compute_Proposition1!(FO)
    compute_BB!(FO)
    ω̄ = reshape(OCM.ω, :, 2)
    p̄ = ω̄ ./ sum(ω̄, dims=2)
    p̄[isnan.(p̄[:, 1]), 1] .= 1.0
    p̄[isnan.(p̄[:, 2]), 2] .= 0.0
    ω̄_0 = (p̄ .* ω̄_0_base)[:]
    FO.X_0 = [A_0] - ZO.P * ZO.X̄
    FO.Θ_0 = [0.0]
    FO.Δ_0 = ω̄_0 - ZO.ω̄
    solve_Xt!(FO)
    Xpath = [ZO.X̄ ZO.X̄ .+ FO.X̂t]
    return Xpath
end


# === Simulation Setup ===

# --- Old steady state ---
OCM_ = OCModel()
r_val,tr_val = 0.041634407732543365,0.6276923506074257
assign!(OCM_, r_val, tr_val)
inputs_ = construct_inputs(OCM_)

# Extract initial asset distribution and capital
X̄_ = getX(OCM_)[[1,3,6]]
A_0 = X̄_[3][1]                       # Initial capital stock
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
p1 = plot(df.t, df.R, ylabel="R", label="")
p2 = plot(df.t, df.Tr, ylabel="Tr", label="")
plot(p1, p2, layout=(2, 1), size=(800, 600), legend=:topright)
