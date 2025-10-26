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
        βEE*λᵉ - u_c,                                           # (3) Euler equation
        u + βV*vᵉ - v,                                         # (4) Bellman equation
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
    @unpack ab_bor_cutoff, ab_col_cutoff, τb,βEE, βV,σ, α_b, ν, χ, a̲, δ , τc, τw, γ, k_min = OCM
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
        βEE*λᵉ - u_c,                                                        # (3) Euler equation
        u + βV*vᵉ - v,                                                      # (4) Value function
        mpn - W,                                                           # (5) Labor FOC
        nb - n,                                                            # (6) Consistency: hired labor = choice
        mpk - r - δ,                                                       # (7) Capital FOC
        yb - z*k^α_b*n^ν,                                                  # (8) Output production function
        profit - (yb - δ*k - r*k - W*n),                                   # (9) Profit definition
        b - 1                                                              # (10) Occupational identity
    ]

    if a_ <= ab_bor_cutoff[lθ]
        ret[3] = a̲ - a
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

    λ⁻ = x⁻[9]; v⁻ = x⁻[10]
    λ⁺ = x⁺[9]; v⁺ = x⁺[10]

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

   
    return [af,nf,kf,yf,nbf,cf,πf,Ibf,λf,vf] #return xf
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
    nθ,nsp,nΩ = size(πθ,1),length(a_sp),length(a_Ω)
    aθ_sp = hcat(kron(ones(nθ),a_sp),kron(lθ,ones(nsp)))
    aθc_sp = [aθ_sp ones(size(aθ_sp,1));aθ_sp 2*ones(size(aθ_sp,1))]
    aθ_Ω = hcat(kron(ones(nθ),a_Ω),kron(lθ,ones(nΩ)))
    aθc_Ω = [aθ_Ω ones(size(aθ_Ω,1));aθ_Ω 2*ones(size(aθ_Ω,1))]

    ℵ = Int[]
    mask = OCM.ω .> 1e-10

    return aknots,OCM.so,aθc_sp,aθc_Ω,ℵ
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
    Ix̄_0 = ZO_0.x̄*ZO_0.Φ*ZO_0.ω̄
    return inputs_0, X̄_0, Ix̄_0,A_0, Taub_0, ω̄_0_base
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

function compute_FOSOpaths(X̄_0,Ix̄_0,A_0, Taub_0, ω̄_0_base, OCM_new)
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



    println("→ Constructing FO Ix paths...")
    compute_x̂t_ω̂t!(FO)
    IX̂=compute_Ixt(FO)
    Ix̄ = ZO.x̄*ZO.Φ*ZO.ω̄
    IxpathFO=[Ix̄_0 Ix̄.+IX̂]
    println("...done")


    # === Compute SO Transition Path ===
    println("→computing SO transition path (bulk of the calclations)...")
    SO = SecondOrderApproximation(FO=FO)
    SO.X_02 = FO.X_0
    SO.Θ_02 = FO.Θ_0
    SO.ω̂k =  FO.ω̂t
    SO.ω̂ak =  FO.ω̂at
    SO.x̂k =  FO.x̂t
    SO.ŷk =  FO.ŷt
    SO.κ̂k =  FO.κ̂t
    SO.X̂k =  FO.X̂t
    compute_Lemma2_ZZ!(SO)
    compute_lemma3_components!(SO)
    compute_ŷtk!(SO)
    compute_lemma3_ZZ!(SO)
    compute_lemma3_ZZ_kink!(SO)
    compute_Lemma4_ZZ!(SO)
    construct_Laa!(SO)
    compute_Corollary2_ZZ!(SO)
    compute_XZZ!(SO)
    println("....done")
    
    # === Collect Results ===
    println("→ Constructing X paths and value function...")
    XpathFO = [X̄_0 ZO.X̄ .+ FO.X̂t]
    XpathSO = [X̄_0 ZO.X̄ .+ FO.X̂t .+ 0.5*SO.X̂tk]
    VinitFO = XpathFO[inputs.Xlab .== :V,2][1] #ZO.X̄[inputs.Xlab .== :V] + FO.X̂t[inputs.Xlab .== :V, 1]
    VinitSO = XpathSO[inputs.Xlab .== :V,2][1]

    println("...done ✅")


    
    return XpathSO, IxpathFO,inputs, VinitSO
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
    _, X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old = setup_old_steady_state!(OCM_old)
    println("Old steady state setup complete.")

    println("Setting up new steady state with τb = $τb_val (takes a few minutes)...")
    OCM_new, _ = setup_new_steady_state(τb_val, OCM_old.τw, OCM_old)
    println("New steady state setup complete.")

    # --- FAST TRANSITION ---
    OCM_new.ρ_τ = ρ_τ_val_fast
    println("Performing transition analysis with ρ_τ = $ρ_τ_val_fast...")
    Xpath, Ixpath, inputs, _ = compute_FOSOpaths(X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old, OCM_new)
    df_transition_fast = save_stuff(Xpath, Ixpath, inputs)
    println("Add residuals to DataFrame...")
    getResiduals!(df_transition_fast, OCM_old, OCM_new)


    CSV.write(filenamefast, df_transition_fast)
    println("Results saved to $(filenamefast)")
    println("Transition analysis complete.")


    # --- SLOW TRANSITION ---
    OCM_new.ρ_τ = ρ_τ_val_slow
    println("Performing transition analysis with ρ_τ = $ρ_τ_val_slow...")
    Xpath, Ixpath, inputs, _ = compute_FOSOpaths(X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old, OCM_new)
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
