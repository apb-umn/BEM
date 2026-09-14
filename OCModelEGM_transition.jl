# NOTE (approximation rewrite, Sep 2026): the zeroth/first/second-order code
# follows the z-variable + discrete-choice formulation (see
# ApproximationChanges.tex).  The second order is AD-only (nested duals) and
# carries no σσ (aggregate-risk) terms — the transition experiment is
# deterministic.
include("SecondOrderApproximation.jl")
 

"""
    Fw(OCM::OCModel, lθ, a_, x, X, yᵉ)

Equilibrium conditions for a worker. This function returns residuals for the system of
equations characterizing a worker's optimization problem, including consumption,
savings, labor, and value function conditions.

"""
function Fw(OCM::OCModel, lθ, a_, x, X, yᵉ)
    @unpack aw_bor_cutoff, βEE, βV, σ, a̲, τc, τw, γ = OCM
    a_ = a_[1]
    a, v, n, k, yb, nb, c, profit, b, λ = x   # xlab order (z-variables a, v first)
    λᵉ, vᵉ = yᵉ
    R, W, Tr,Taub = X
    _, ϵ = exp.(lθ)

    # Marginal utility and utility
    u_c = c^(-σ)
    u = c^(1 - σ) / (1 - σ)

    # FOC system.  ROW ORDER (z-formulation): the two forward-looking
    # equations that determine the z-variables (a, v) come FIRST, followed by
    # the intra-temporal equations that determine y = (n,k,yb,nb,c,profit,b,λ).
    ret = [
        βEE*λᵉ - u_c,                                         # (1) Euler equation            [z: a]
        u + βV*vᵉ - v,                                        # (2) Bellman equation          [z: v]
        R*a_ + (1 - τw)*W*ϵ + Tr - (1 + τc)*c - (1 + γ)*a,    # (3) Budget constraint         [y: c]
        λ - R*u_c,                                            # (4) Envelope condition: ∂V/∂a [y: λ]
        n + ϵ,                                                # (5) Labor supply condition (ϵ = disutility shock)
        k, yb, nb, profit, b                                  # (6–10) unused in worker case
    ]

    # Replace Euler with borrowing constraint if a_ below cutoff
    if a_ <= aw_bor_cutoff[lθ]
        ret[1] = a̲ - a  # (1b) Binding borrowing constraint
    end

    return ret
end




"""
    Fb(OCM::OCModel, lθ, a_, x, X, yᵉ)

Equilibrium conditions for a business owner. Includes budget constraint,
first-order conditions for inputs, and collateral/borrowing constraints.
"""
function Fb(OCM::OCModel, lθ, a_, x, X, yᵉ)
    @unpack ab_bor_cutoff, ab_col_cutoff, τb,βEE, βV,σ, α_b, ν, χ, a̲, δ , τc, τw, γ, k_min = OCM
    a_ = a_[1]
    a, v, n, k, yb, nb, c, profit, b, λ = x   # xlab order (z-variables a, v first)
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

    # System of residuals.  ROW ORDER (z-formulation): forward-looking
    # equations for the z-variables (a, v) first, intra-temporal ones after.
    ret = [
        βEE*λᵉ - u_c,                                                        # (1) Euler equation            [z: a]
        u + βV*vᵉ - v,                                                      # (2) Value function            [z: v]
        R*a_ + (1 - Taub)*profit + Tr - (1 + τc)*c - (1 + γ)*a,            # (3) Budget constraint         [y: c]
        λ - R*u_c - u_c*χ*(mpk - r - δ)*(1-Taub),                          # (4) Marginal value of wealth  [y: λ]
        mpn - W,                                                           # (5) Labor FOC
        nb - n,                                                            # (6) Consistency: hired labor = choice
        mpk - r - δ,                                                       # (7) Capital FOC
        yb - z*k^α_b*n^ν,                                                  # (8) Output production function
        profit - (yb - δ*k - r*k - W*n),                                   # (9) Profit definition
        b - 1                                                              # (10) Occupational identity
    ]

    if a_ <= ab_bor_cutoff[lθ]
        ret[1] = a̲ - a
        ret[7] = k - (χ*a_ + k_min)
    elseif (a_ <= ab_col_cutoff[lθ]) && (a_ > 0)
        ret[7] = k - (χ*a_ + k_min)
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
    Ia, Iv, In, Ik, Iyb, Inb, Ic, Iprofit, Ib, _ = Ix   # xlab order (a, v first; Iλ unused)
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

    λ⁻ = x⁻[10]; v⁻ = x⁻[2]     # xlab = [a, v, n, k, yb, nb, c, profit, b, λ]
    λ⁺ = x⁺[10]; v⁺ = x⁺[2]

    Δv = (v⁺ - v⁻) / σ_ε
    T = promote_type(typeof(λ⁻), typeof(λ⁺), typeof(σ_ε), typeof(Δv))
    oneT = one(T)
    p = oneT / (oneT + exp(Δv))

    if p < T(1e-9)
        return T[λ⁺, v⁺]  # Ensure return type is Vector{T}
    elseif p > T(1 - 1e-9)
        return T[λ⁻, v⁻]
    else
        Ev = v⁻ + σ_ε * log1p(exp(Δv))
        return T[λ⁻ * p + λ⁺ * (1 - p), Ev]
    end
end


"""
    pf(para::OCModel, x⁻, x⁺)

Probability of choosing occupation 1 (worker) given the two occupations'
values at the same (a,θ): Prob(c = 1) = 1/(1 + exp((v⁺ − v⁻)/σ_ε)).  Must be
consistent with the choice probability inside `ff`.
"""
function pf(para::OCModel, x⁻, x⁺)
    @unpack σ_ε = para
    v⁻ = x⁻[2]; v⁺ = x⁺[2]
    return 1 / (1 + exp((v⁺ - v⁻) / σ_ε))
end


 """
    save_policy_functions!(OCM::OCModel)

Saves the policy functions in the OCModel object
"""

function get_policy_functions(OCM::OCModel)
    @unpack bf,wf,curv_a,Na,amax,a̲,curv_h,Ia,r,σ,δ,α_b,agrid,Nθ,lθ,ν,χ,τb=OCM

    #save the policy functions a,n,k,λ,v
    af(lθ,a,c) = c==1 ? wf.a[[lθ].==eachrow(OCM.lθ)][1](a) : bf.a[[lθ].==eachrow(OCM.lθ)][1](a)
    nf(lθ,a,c) = c==1 ? -exp.(lθ[2]) : bf.n[[lθ].==eachrow(OCM.lθ)][1](a)
    kf(lθ,a,c) = c==1 ? 0 : bf.k[[lθ].==eachrow(OCM.lθ)][1](a)
    yf(lθ,a,c) = c==1 ? 0 : bf.y[[lθ].==eachrow(OCM.lθ)][1](a)
    nbf(lθ,a,c) = c==1 ? 0 : bf.n[[lθ].==eachrow(OCM.lθ)][1](a)
    cf(lθ,a,c) = c==1 ? wf.c[[lθ].==eachrow(OCM.lθ)][1](a) : bf.c[[lθ].==eachrow(OCM.lθ)][1](a)
    λf(lθ,a,c) = c==1 ? wf.λ[[lθ].==eachrow(OCM.lθ)][1](a) : bf.λ[[lθ].==eachrow(OCM.lθ)][1](a)
    vf(lθ,a,c) = c==1 ? wf.v[[lθ].==eachrow(OCM.lθ)][1](a) : bf.v[[lθ].==eachrow(OCM.lθ)][1](a)
    πf(lθ,a,c) = c==1 ? 0 : bf.π[[lθ].==eachrow(OCM.lθ)][1](a)
    Ibf(lθ,a,c) = c==1 ? 0 : 1

   
    return [af,vf,nf,kf,yf,nbf,cf,πf,Ibf,λf] #return xf in xlab order (z-variables a, v first)
end




function get_grids(OCM)
    @unpack bf,wf,curv_a,Na,amax,a̲,curv_h,Ia,πθ,lθ=OCM
    xvec = LinRange(0,1,Na-1).^curv_a  #The Na -1 to adjust for the quadratic splines
    âgrid = a̲ .+ (amax - a̲).*xvec #nonlinear grid for knot points
    xvec = LinRange(0,1,Ia).^curv_h 
    āgrid = a̲ .+ (amax - a̲).*xvec #nonlinear grids for distribution
    aknots = [âgrid]
    a_sp = nodes(SplineParams(aknots[1],0,OCM.so)) #construct gridpoints from knots
    a_Ω = āgrid
    # REDUCED (a,θ) tensor grids in the canonical layout (a fastest, θ slowest);
    # the policy grid is the sparse grid replicated for c = 1,2 (c slowest)
    aθ_sp = tensor_grid([collect(a_sp)], lθ)
    aθ_Ω  = tensor_grid([collect(a_Ω)], lθ)
    ℵ = Int[]
    return aθ_sp, aθ_Ω, ℵ
end



function getX(OCM::OCModel)
   @unpack r,tr,w,b = OCM 
   cdst,adst,vdst,_,_,_,_ = dist!(OCM)

   R=r+1 # gross interest rate
   W=w # wage rate
   T=tr # transfer
   Frac_b =sum(reshape(OCM.ω,:,2),dims=1)[2] # fraction of borrowing agents
   V = dot(OCM.ω,vdst) #average utility
   A = dot(OCM.ω,adst) # average assets
   C      = dot(OCM.ω,cdst) # average consumption
   X̄ = [R,W,T,Frac_b,V,A,C]
   return X̄ 
end



"""
    get_conditional_transitions(OCM)

The steady-state transition operators on the REDUCED fine grid (a,θ)
CONDITIONAL on the occupation, Λc[c] (c = 1 worker, c = 2 owner), built from
the same ingredients as `dist!`: the linear (lottery) interpolation of the
savings policy a′(a,θ,c) onto the histogram grid, per θ block, composed with
the Markov mixing πθ.  The unconditional pre-choice transition is
Λ̃ = Σ_c Λc[c]·diag(p̄_c) (formed inside ZerothOrderApproximation), and
Λ̃ μ̄ = μ̄ holds exactly for μ̄ = Σ_c OCM.ω_c.
Also returns p̄, the steady-state probability of being a worker on the fine grid.
"""
function get_conditional_transitions(OCM::OCModel)
    @unpack wf,bf,Nθ,πθ,Ia,alθ,σ_ε = OCM
    ah  = alθ[1:Ia,1]
    afw = max.(min.(hcat([wf.a[s](ah) for s in 1:Nθ]...),ah[end]),ah[1])
    afb = max.(min.(hcat([bf.a[s](ah) for s in 1:Nθ]...),ah[end]),ah[1])
    Vw  = hcat([wf.v[s](ah) for s in 1:Nθ]...)
    Vb  = hcat([bf.v[s](ah) for s in 1:Nθ]...)
    p̄  = probw.(Vb.-Vw,σ_ε)[:]
    B   = Basis(SplineParams(ah,0,1))
    Aw  = [sparse(BasisMatrix(B,Direct(),@view afw[:,s]).vals[1]') for s in 1:Nθ]
    Ab  = [sparse(BasisMatrix(B,Direct(),@view afb[:,s]).vals[1]') for s in 1:Nθ]
    Π   = Matrix(transpose(πθ))
    Λc  = [FactoredTransitionMatrix(Φs=Aw, Π=Π), FactoredTransitionMatrix(Φs=Ab, Π=Π)]
    return Λc, p̄
end


"""
    construct_inputs(OCM)

Create and return `Inputs` object with model functions, grids, and equilibrium mappings.
Used to compute steady state and approximations.
"""
function construct_inputs(OCM)
    inputs = Inputs()

    # Labels (z-variables FIRST: the states, then the forward-looking value)
    inputs.xlab  = [:a, :v, :n, :k, :yb, :nb, :c, :profit, :b, :λ]
    inputs.alab  = [:a]
    inputs.zlab  = [:a, :v]
    inputs.xelab = [:λ, :v]          # rows read by f (=ff) and pf
    inputs.yᵉlab = [:λᵉ, :vᵉ]        # outputs of f whose expectations enter F
    inputs.Ilab  = [:a, :v, :n, :k, :yb, :nb, :c, :profit, :b]   # integrals G reads (all but λ)

    # Grids and policy VALUES at the full sparse nodes (c slowest)
    xf = get_policy_functions(OCM)
    inputs.aθ_sp, inputs.aθ_Ω, inputs.ℵ = get_grids(OCM)
    nspr = size(inputs.aθ_sp,1)
    na = length(inputs.alab)
    nx = length(xf)
    x̄ = zeros(nx, 2*nspr)
    for c in 1:2, jr in 1:nspr
        a = inputs.aθ_sp[jr, 1:na]
        θ = inputs.aθ_sp[jr, na+1:end]
        for ix in 1:nx
            x̄[ix, jr + (c-1)*nspr] = xf[ix](θ, a, c)[1]
        end
    end
    inputs.x̄ = x̄

    # Aggregates and equilibrium labels
    inputs.X̄ = [getX(OCM);OCM.τb]
    inputs.Xlab = [:R, :W, :Tr, :Frac_b, :V, :A, :C,:Taub]
    inputs.Alab = [:A,:Taub]
    inputs.Qlab = [:R, :W, :Tr,:Taub]

    # Distributional objects: pre-choice masses μ̄ = Σ_c ω̄_c, worker
    # probability p̄ and the conditional transitions on the reduced fine grid
    inputs.μ̄ = vec(sum(reshape(OCM.ω, :, 2), dims=2))
    inputs.Λc, inputs.p̄ = get_conditional_transitions(OCM)
    inputs.πθ = OCM.πθ
    inputs.Θ̄ = ones(1) * OCM.Θ̄
    inputs.ρ_Θ = ones(1, 1) * 0.8
    inputs.Σ_Θ = ones(1, 1) * 0.017^2

    # Residual functions (F receives the full node index j and the occupation c)
    inputs.para = OCM
    inputs.F = (j, lθ, a_, c, x, X, yᵉ) -> c == 1 ? Fw(OCM, lθ, a_, x, X, yᵉ) : Fb(OCM, lθ, a_, x, X, yᵉ)
    inputs.G = (Ix, X_, X, Xᵉ, lΘ) -> G(OCM, Ix, X_, X, Xᵉ, lΘ)
    inputs.f = (x⁻, x⁺) -> ff(OCM, x⁻, x⁺)
    inputs.pf = (x⁻, x⁺) -> pf(OCM, x⁻, x⁺)

    return inputs
end




function setup_old_steady_state!(OCM)
    #OCM.ibise = 0 
    OCM.iprint = 0
    solvess!(OCM)
    updatecutoffs!(OCM)
    inputs_0 = construct_inputs(OCM)
    X̄_0 = [getX(OCM); OCM.τb]
    A_0 = X̄_0[inputs_0.Xlab .== :A][1]
    Taub_0 = X̄_0[inputs_0.Xlab .== :Taub][1]
    μ̄_0 = inputs_0.μ̄                       # pre-choice masses of the old steady state
    ZO_0 = ZerothOrderApproximation(inputs_0)
    Ix̄_0 = ZO_0.x̄*(ZO_0.Φ*ZO_0.ω̄)
    return inputs_0, X̄_0, Ix̄_0,A_0, Taub_0, μ̄_0
end

function setup_new_steady_state(τb, τw, OCM_old)
    OCM = deepcopy(OCM_old)
    OCM.τb = τb
    OCM.τw = τw
    OCM.iprint = 0
    OCM.ibise = 1
    assign!(OCM, OCM_old.r, OCM_old.tr)
    ss, lev, shr, res = solvess!(OCM)
    updatecutoffs!(OCM)
    Xss = [getX(OCM); OCM.τb]
    return OCM, Xss
end

"""
    compute_FOpaths(X̄_0, Ix̄_0, A_0, Taub_0, μ̄_0, OCM_new; check=true)

First-order transition path from the old steady state (aggregates X̄_0,
integrals Ix̄_0, predetermined A_0, Taub_0 and pre-choice distribution μ̄_0)
to the new steady state described by OCM_new.  Returns
(XpathFO, IxpathFO, inputs, VinitFO, FO).  With `check=true` the zeroth-order
residuals and the J-matrix consistency of the paths are printed.
"""
function compute_FOpaths(X̄_0,Ix̄_0,A_0, Taub_0, μ̄_0, OCM_new; check::Bool=true)
    println("→ Constructing inputs...")
    inputs = construct_inputs(OCM_new)
    println("...done")

    println("→ Zeroth-order approximation...")
    ZO = ZerothOrderApproximation(inputs)
    check && println("   stationarity residual ‖Λ̃μ̄ − μ̄‖∞ = ", stationarity_residual(ZO))
    println("...done")

    println("→ Computing derivatives...")
    Fres = computeDerivativesF!(ZO, inputs)
    Gres = computeDerivativesG!(ZO, inputs)
    check && println("   max |F| at steady state = ", maximum(abs, Fres), ",  max |G| = ", maximum(abs, Gres))
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

    println("→ Setting initial conditions...")
    FO.X_0 = [A_0; Taub_0] - ZO.P * ZO.X̄
    FO.Θ_0 = [0.0]
    FO.Ω_0 = μ̄_0 - ZO.μ̄                 # initial pre-choice mass perturbation
    println("...done")

    println("→ Solving transition path...")
    solve_Xt!(FO)
    println("...done")

    println("→ Constructing FO Ix paths...")
    compute_x̂t_Ω̂t!(FO)
    IX̂=compute_Ixt(FO)
    Ix̄ = ZO.x̄*(ZO.Φ*ZO.ω̄)
    IxpathFO=[Ix̄_0 Ix̄.+IX̂]
    check && println("   J-matrix consistency of the Ix paths (rel.) = ", check_J_consistency(FO))
    println("...done")

    XpathFO = [X̄_0 ZO.X̄ .+ FO.X̂t]
    VinitFO = XpathFO[inputs.Xlab .== :V,2][1]
    println("...done ✅")
    return XpathFO, IxpathFO, inputs, VinitFO, FO
end


"""
    compute_FOSOpaths(X̄_0, Ix̄_0, A_0, Taub_0, μ̄_0, OCM_new; check=true)

First- and second-order transition paths.  The second-order correction is the
(t,k) interaction of the transition direction with itself (initial conditions
X_0, μ̂_1 = μ̄_0 − μ̄; no aggregate shocks), computed by the AD-only chain of
SecondOrderApproximation.jl.  Returns (XpathSO, IxpathFO, inputs, VinitSO)
as before, plus XpathFO and the SO object:
    XpathSO, IxpathFO, inputs, VinitSO, XpathFO, SO
"""
function compute_FOSOpaths(X̄_0,Ix̄_0,A_0, Taub_0, μ̄_0, OCM_new; check::Bool=true)
    XpathFO, IxpathFO, inputs, VinitFO, FO = compute_FOpaths(X̄_0,Ix̄_0,A_0, Taub_0, μ̄_0, OCM_new; check=check)
    ZO = FO.ZO

    # === Compute SO Transition Path ===
    println("→computing SO transition path (bulk of the calclations)...")
    SO = SecondOrderApproximation(FO=FO)
    SO.X_02 = FO.X_0
    SO.Θ_02 = FO.Θ_0
    SO.Ω̂k = FO.Ω̂t
    SO.ẑk = FO.ẑt
    SO.X̂k = FO.X̂t
    compute_ZZ_transition!(SO, inputs)
    println("....done")

    # === Collect Results ===
    println("→ Constructing X paths and value function...")
    XpathSO = [X̄_0 ZO.X̄ .+ FO.X̂t .+ 0.5*SO.X̂tk]
    VinitSO = XpathSO[inputs.Xlab .== :V,2][1]
    println("...done ✅")

    return XpathSO, IxpathFO, inputs, VinitSO, XpathFO, SO
end

function getResiduals!(df,OCM_old, OCM_new)
    T = size(df, 1) - 1
    rT = df.R[2:T+1] .- 1
    trT = df.Tr[2:T+1]
    τbT = df.Taub[2:T+1]
    x0 = vcat(rT, trT)
    OCM_old.T = T
    OCM_new.T = T
    res = residual_tr!(x0, OCM_old, OCM_new,τbT)
    assetmarketres= reshape(res,(T,2))[:,1]
    gbcres= reshape(res,(T,2))[:,2]
    df[!,:AssetMarketResidual] = vcat(0.0, assetmarketres)
    df[!,:GBCResidual] = vcat(0.0, gbcres)
    println("Asset Market Residual: ", norm(assetmarketres))
    println("Government Budget Constraint Residual: ", norm(gbcres))
    return res
end

function save_stuff(Xpath, Ixpath, inputs)

    df = DataFrame(Xpath', inputs.Xlab)
    df.t = 0:(size(Xpath, 2) - 1)

    df=hcat(df, DataFrame(Ixpath', map(x -> Symbol("I", String(x)), inputs.xlab)))


    return df
end



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
        "Transfers (Tr)", "Wage (W)", "Business Tax (Taub)"
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




"""
    run_transition_analysis(τb_val, ρ_τ_val_fast, ρ_τ_val_slow, filenamefast, filenameslow, saveplotfilename)

Run and compare two transition analyses — one with a fast adjustment and one with a slow adjustment of the capital income tax `τb`.

# Arguments
- `τb_val::Float64`: The value of τb (capital tax) in the new steady state.
- `ρ_τ_val_fast::Float64`: Transition speed for the fast reform (ρ_τ close to 0).
- `ρ_τ_val_slow::Float64`: Transition speed for the slow reform (ρ_τ close to 1).
- `filenamefast::String`: Filename to save the transition path for the fast adjustment.
- `filenameslow::String`: Filename to save the transition path for the slow adjustment.
- `saveplotfilename::String`: Filename to save the comparison plot of transition paths.

# Returns
- `df_transition_fast::DataFrame`: Transition path data for the fast adjustment.
- `df_transition_slow::DataFrame`: Transition path data for the slow adjustment.

# Workflow
1. Constructs the old steady state using baseline values.
2. Solves the new steady state under `τb_val`.
3. Simulates transition dynamics for both fast and slow reforms.
4. Saves the transition paths (`df_transition_fast` and `df_transition_slow`) to CSV.
5. Plots key variables comparing fast vs. slow reform paths and saves the plot.

"""
function run_transition_analysis(
    τb_val,
    ρ_τ_val_fast,
    ρ_τ_val_slow,
    filenamefast,
    filenameslow,
    saveplotfilename;
    r::Float64 = 0.040063534877869074,   # default interest rate
    tr::Float64 = 0.9470530702667457     # default transfer level
)
    println("Setting up old steady state (takes a few minutes)...")

    OCM_old = OCModel()
    setup!(OCM_old)

    # assign parameters (using provided or default values)
    OCM_old.r = r
    OCM_old.tr = tr 
    _, X̄_old, Ix̄_old, A_old, Taub_old, μ̄_old = setup_old_steady_state!(OCM_old)
    println("Old steady state setup complete.")

    println("Setting up new steady state with τb = $τb_val (takes a few minutes)...")
    OCM_new, _ = setup_new_steady_state(τb_val, OCM_old.τw, OCM_old)
    println("New steady state setup complete.")

    # --- FAST TRANSITION ---
    OCM_new.ρ_τ = ρ_τ_val_fast
    println("Performing transition analysis with ρ_τ = $ρ_τ_val_fast...")
    Xpath, Ixpath, inputs, _ = compute_FOSOpaths(X̄_old, Ix̄_old, A_old, Taub_old, μ̄_old, OCM_new)
    df_transition_fast = save_stuff(Xpath, Ixpath, inputs)
    println("Add residuals to DataFrame...")
    getResiduals!(df_transition_fast, OCM_old, OCM_new)


    CSV.write(filenamefast, df_transition_fast)
    println("Results saved to $(filenamefast)")
    println("Transition analysis complete.")


    # --- SLOW TRANSITION ---
    OCM_new.ρ_τ = ρ_τ_val_slow
    println("Performing transition analysis with ρ_τ = $ρ_τ_val_slow...")
    Xpath, Ixpath, inputs, _ = compute_FOSOpaths(X̄_old, Ix̄_old, A_old, Taub_old, μ̄_old, OCM_new)
    df_transition_slow = save_stuff(Xpath, Ixpath, inputs)

    println("Add residuals to DataFrame...")
    getResiduals!(df_transition_slow, OCM_old, OCM_new)
    
    CSV.write(filenameslow, df_transition_slow)
    println("Results saved to $(filenameslow)")


 
    # --- Plotting ---
    plot_transition_comparison_dfs(df_transition_slow, df_transition_fast, savepath=saveplotfilename)
    println("Transition comparison plot saved to $(saveplotfilename)")

    println("Transition analysis completed successfully.")

    return df_transition_fast, df_transition_slow
end
