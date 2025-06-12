

include("OCModelE.jl")
include("FirstOrderApproximation.jl")
 

"""
    Fw(OCM::OCModel, lθ, a_, x, X, yᵉ)

Equilibrium conditions for a worker. This function returns residuals for the system of
equations characterizing a worker's optimization problem, including consumption,
savings, labor, and value function conditions.

"""
function Fw(OCM::OCModel, lθ, a_, x, X, yᵉ)
    @unpack aw_bor_cutoff, β, σ, a̲, τc, τw, γ, α, Θ̄, τp, δ = OCM
    a_ = a_[1]
    a,â_, n, k, yb, nb, c, profit, b, λ, v = x
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
        k, yb, nb, profit, b,                               # (6–10) Unused
        â_ - a_                                           #track previous period's assets
    ]
    if a_ <= aw_bor_cutoff[lθ]
        ret[3] = a̲ - a  # (3b) Borrowing constraint binds
    end
    return ret
end




"""
    Fb(OCM::OCModel, lθ, a_, x, X, yᵉ)

Equilibrium conditions for a business owner. Includes budget constraint,
first-order conditions for inputs, and collateral/borrowing constraints.
"""
function Fb(OCM::OCModel, lθ, a_, x, X, yᵉ)
    @unpack ab_bor_cutoff, ab_col_cutoff, β, σ, α_b, ν, χ, a̲, τb, τc, τw, γ, Θ̄, τp, δ, α = OCM
    a_ = a_[1]
    a,â_, n, k, yb, nb, c, profit, b, λ, v = x
    λᵉ, vᵉ = yᵉ
    R, Tr = X
    r = R - 1
    z, _ = exp.(lθ)
    # Marginal products
    
    W = (1 - α) * Θ̄ * ((r / (1 - τp) + δ) / (Θ̄ * α))^(α / (α - 1))
    mpk = α_b * z * k^(α_b - 1) * n^ν
    mpn = ν * z * k^α_b * n^(ν - 1)

    # Utility
    u_c = c^(-σ)
    u = c^(1 - σ) / (1 - σ)
     # System of residuals
   
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
        b - 1,                                                   # (10)
        â_ - a_                                               #track previous period's assets
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


"""
    G(para::OCModel, Ix, A_, X, Xᵉ, Θ)

Aggregate equilibrium conditions. Ensures market clearing, government budget
balance, and correct pricing in production sectors.
"""
function G(para::OCModel, Ix, A_, X, Xᵉ, Θ)
    @unpack α, δ, τw, τb, τp, τd, τc, b, w, γ, r, g = para

    # === Unpack aggregate integrals ===
    Ia,Ia_, In, Ik, Iyb, Inb, Ic, Iprofit, Ib, _, Iv = Ix
    R, Tr = X
    TFP = Θ[1]               # Current TFP
    #A_ = Ia_               # Lagged aggregate assets
    B = b                    # Government debt
    r = R - 1.0              # Interest rate on capital
    Rc = (R - 1) / (1 - τp) + 1

    # === Labor and capital in corporate and business sectors ===
    Nc = -In                 # Corporate labor (In is negative by convention)
    Nb = Inb                 # Business labor
    Kb = Ik                  # Business capital
    #Kc = (A_ - Ik - B) / (1 - τd)   #  corporate capital (from assets)
    #MPKc = α * TFP * Kc^(α - 1) * Nc^(1 - α)

    #Solve for Kc from MPKc
    MPKc = Rc - 1 + δ
    Kc = (MPKc / (α * TFP))^(1 / (α - 1)) * Nc 
    A_ = Kc + Ik + B

    # === Output in each sector ===
    Yc = TFP * Kc^α * Nc^(1 - α)    # Corporate output
    Yb = Iyb                         # Business output (from integral)

    #1-(tx-trT[t]-(rT[t]-γ)*b)/g

    # === Government tax revenues ===
    Tp = τp * (Yc - w * Nc - δ * Kc)                    # Capital income tax
    Td = τd * (Yc - w * Nc - (γ + δ) * Kc - Tp)         # Dividend tax
    Tn = τw * w * (Nc + Nb)                             # Labor income tax
    Tb = τb * (Yb - (r + δ) * Kb - w * Nb)              # Business profit tax
    Tx=τc * Ic + Tp + Td + Tn + Tb
    # === General equilibrium residuals ===
    return [
        1 - Ia_/A_,  # (1) Asset market clearing: actual assets vs. distribution-integrated assets
        1-(Tx- B * (R - 1 - γ)  - Tr)/g  # (2) Government budget constraint
    ]
end



"""
    ff(para::OCModel, x⁻, x⁺)

Smooth discrete occupational choice via log-sum (softmax).
Returns expected marginal utility and expected value.
"""
function ff(para::OCModel, x⁻, x⁺)
    @unpack σ_ε = para
    _,_,_,_,_,_,_,_,_,λ⁻,v⁻ = x⁻
    _,_,_,_,_,_,_,_,_,λ⁺,v⁺ = x⁺
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


"""
    construct_inputs(OCM)

Create and return `Inputs` object with model functions, grids, and equilibrium mappings.
Used to compute steady state and approximations.
"""
function construct_inputs(OCM)
    inputs = Inputs()
    inputs.xf = get_policy_functions_with_â(OCM) 
    inputs.aknots, inputs.ka, inputs.aθc_sp, inputs.aθc_Ω, inputs.ℵ = get_grids(OCM)
    inputs.xlab = [:a, :â_, :n, :k, :yb, :nb, :c, :profit, :b, :λ, :v]
    inputs.alab = [:a]
    inputs.κlab = [:v]
    inputs.yᵉlab = [:λ, :v]
    inputs.Γf = κ -> 1 / (1 + exp(κ / OCM.σ_ε))
    inputs.dΓf = κ -> -(1 / OCM.σ_ε) * exp(κ / OCM.σ_ε) / (1 + exp(κ / OCM.σ_ε))^2
    inputs.X̄ = getX(OCM)[[1, 3]]
    inputs.Xlab = [:R, :Tr]
    inputs.Alab = [:R]
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


function compute_FO_transition_path(τb_val, τw_val)

    # === Old Steady State ===
    println("setting up the initial ss...")
    OCM_ = OCModel()
    OCM_.iprint = 0
    setup!(OCM_)
    OCM_.ibise = 1
    ss_,_,shr_,res=solvess!(OCM_)
    inputs_ = construct_inputs(OCM_)
    X̄_ = getX(OCM_)[[1,3]]
    A_0 = X̄_[1]                       # Initial capital stock
    ω̄_0_base = sum(reshape(OCM_.ω, :, 2), dims=2)
    println("....done")
    case_ = (τb=OCM_.τb, τw=OCM_.τw, χ=OCM_.χ)
    print_OC_model_summary(ss_, shr_,case_)


    # === New Steady State ===
    println("setting up the new ss...")
    OCM = deepcopy(OCM_)
    OCM.τb = τb_val
    OCM.τw = τw_val
    assign!(OCM, OCM_.r, OCM_.tr)
    OCM.ibise = 0
    OCM.iprint = 0
    ss,_,shr,res= solvess!(OCM)
    diff_v = OCM.diffv
    Xss = getX(OCM)[[1,3]]
    R, Tr = Xss
    println("....done")
    case = (τb=OCM.τb, τw=OCM.τw, χ=OCM.χ)
    print_OC_model_summary(ss, shr,case)


    # === Construct Inputs ===
    println("constructing inputs...")
    inputs = construct_inputs(OCM)
    ZO = ZerothOrderApproximation(inputs)
    computeDerivativesF!(ZO, inputs)
    computeDerivativesG!(ZO, inputs)
    FO = FirstOrderApproximation(ZO, OCM.T)
    println("....done")

    # === Compute FO Transition Path ===
    println("computing FO transition path...")
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
    println("....done")
    Xpath = [X̄_ ZO.X̄ .+ FO.X̂t]

    df = DataFrame(Xpath',inputs.Xlab)
    df.t = 0:(size(Xpath,2)-1)

    jac=FO.BB
    return df,jac,OCM_,OCM
end
