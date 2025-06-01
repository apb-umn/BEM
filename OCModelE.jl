# OCModelE.jl
#
#   Three business sectors:
#     -- Nonfinancial corporate sector:
#
#         vc(Kc) =  max {(1-τd)dc +(1+γ)/(1+r) vc(Kc')}
#                  Nc,Xc 
#
#           s.t. (1+γ)Kc'=(1-δ)Kc+Xc
#                     Yc = Θ̄ Kc^α Nc^(1-α)
#                     dc = Yc-w*Nc-Xc-τp(Yc-w*Nc-δ*Kc)
#
#    -- Financial corporate sector
#         vi(x) =  max  {di+ (1+γ)/(1+r) vi(x')} 
#                   x'    
#
#           s.t. di = q*s+b+∫kb-∫a 
#                      +(1-τd)dc*s+r*b-rp∫kb-r∫a (income)
#                      -(1+γ)[q*s'+b'-∫a']-∫[kb'-(1-δ)kb]
#                   = net worth+income-invetment
#
#        ==> r=rp-δ=(1-τd)*[q+(1-τd)dc]/q[-1]-1
#                =(1-τp)(α*Yc/Kc-δ)
#
#    -- Private sector operating DRS technology 
#
#         vb(a,θw,θb) =  max U(c) + β E v(a',θw',θb',η')
#                        c,a'
#
#           s.t. c+a' = (1+r)a + π - τb*π-τc*c
#                π = max θb kb^α nb^ν-(r+δ)*kb-w*nb
#
#   Workers:
#
#         vw(a,θw,θb) =  max U(c) + β E v(a',θw',θb',η')
#                       c,a'
#
#           s.t. c+a' = (1+r)*a + w*θw -τw*w*θw -τc*c
#
#   Occupation choice:
#
#         η = taste shock = ηw-ηb ~ logistic
#         p = probability of being a worker
#         v = max d*(vw+η) + (1-d)*vb
#
#        Ev = 0.57722*σ_η + σ_η *log(exp(vw/σ_η)+exp(vb/σ_η))
#           = 0.57722*σ_η + vw+ σ_η*(1+exp((vb-vw)/σ_η))
#
#   Market clearing:
#
#        Kc + ∫ (1-d[i])k[i] di = ∫ a[i] di
#        Nc + ∫ (1-d[i])n[i] di = ∫ d[i] θw[i] di
#
#   Government budget:
#
#        G+Tr+r*B =B'-B+τc*C+τw*w*(Nc+Nb)
#                  +τp*(Yc-w*Nc-δ*Kc)
#                  +τd*(Yc-w*Nc-Xc-τp*(Yc-w*Nc-δ*Kc))
#                  +τb*(Yp-(r+δ)*Kp-w*Nb)
#

#   Ellen McGrattan, 5/9/2025
#   Revised, ERM, 5/25/2025

using Parameters,LinearAlgebra,BasisMatrices,SparseArrays,Arpack,Roots, 
      KrylovKit,QuantEcon, PrettyTables,StatsBase,ForwardDiff,Dierckx,
      Plots,NPZ,NLsolve,Printf,DataFrames,CSV,Distances

"""
Parameters of the Occupation Choice Model (Lucas Version)
"""
@with_kw mutable struct OCModel

    #Preferences
    σ::Float64   = 1.5                  #Risk Aversion
    βo::Float64  = 1.00776              #Discount Factor (original)
    γ::Float64   = 0.02                 #Economy growth rate
    β::Float64   = 0.99978311           #Discount Factor (with growth) 
    σ_ε::Float64 = 0.05                 #St.Dev. of taste shock ε

    #Corporate parameters
    α::Float64   = 0.5                  #Corporate capital share
    Θ̄::Float64   = 0.70                  #Corporate TFP
    δ::Float64   = 0.041                #Depreciation rate

    #Entrepreneur parameters
    α_b::Float64 = 0.33                 #Private capital share
    ν::Float64   = 0.33                 #Private labor share 
    χ::Float64   = 2.0                  #Collateral constraint

    #Entrepreneur income shocks
    N_θb::Int     = 5                   #Number of productivity shocks θb
    ρ_θb::Float64 = 0.966               #Persistence of θb
    σ_θb::Float64 = 0.20                #St. Dev. of θb
    bm_θb::Int    = 1                   #Use estimates from BM (2021)

    #Worker income shocks 
    N_θw::Int     = 5                   #Number of productivity shocks θw
    ρ_θw::Float64 = 0.966               #Persistence of θw
    σ_θw::Float64 = 0.13                #St.Dev. of θw
    bm_θw::Int    = 1                   #Use estimates from BM (2021)

    #Asset grids
    a̲::Float64    = 0.0                 #Borrowing constraint
    amax::Float64 = 150.             #Maximum asset grid point
    Na::Int       = 50                  #Number of gridpoints for splines
    so::Int       = 2                   #Spline order for asset grid
    Ia::Int       = 1000                #Number of gridpoints for histogram
    curv_a::Float64 = 3.0               #Controls spacing for asset grid
    curv_h::Float64 = 3.0               #Controls spacing for histogram

    #Fiscal policy
    g::Float64    = 0.15                #Government spending on G&S
    b::Float64    = 3.0                 #Debt
    τb::Float64   = 0.2                 #Tax on private business
    τw::Float64   = 0.4                 #Tax on wages
    τp::Float64   = 0.36                #Tax on corporate profits
    τd::Float64   = 0.133               #Tax on dividend
    τc::Float64   = 0.065               #Tax on consumption
    tx::Float64   = 0.0                 #Total tax

    #Numerical parameters
    trlb::Float64 = 0.2                 #Transfers lower bound
    trub::Float64 = 1.0                 #Transfers upper bound
    Ntr::Int      = 1                   #Number of transfer evaluations
    rlb::Float64  = 0.0375               #Rate lower bound 
    rub::Float64  = 0.045               #Rate upper bound 
    Nr::Int       = 3                   #Number of rate evaluations (in check!)
    Neval::Int    = 2                   #Number of bisection evaluations
    iagg::Int     = 1                   #Show aggregate data for each r/tr combo
    λ::Float64    = 1.0                 #Weight on Vcoefs update
    Nit::Int      = 1000                #Number of iterations in solve_eg!
    tolegm::Float64 = 1e-5            #Tolerance for EGM convergence
    ftolmk::Float64 = 1e-6            #Tolerance for market clearing
    xtolxmk::Float64 = 1e-3            #Tolerance for market clearing
    maxitermk::Int = 1000         #Maximum iterations for market clearing
    ibise::Int = 1              #Bisection method for initial guess of r/tr
    iprint::Int   = 1                   #Turn on=1/off=0 intermediate printing
    T::Int        = 300                 #Number of periods in solve_tr!
    ξ::Float64    = 1.0                 #Newton relaxation parameter
    inewt::Int    = 1                   #Use simple Newton in solvess

    #Vector/matrix lengths
    Nθ::Int       = N_θb*N_θw           #Total number of income shocks
    Nv::Int       = Nθ*Na               #Length of V coefficients
    Nh::Int       = Nθ*Ia*2             #Total number of histogram states

    #
    # Values to be filled in setup!
    #   agrid = grid vector for endogenous grid
    #   lθ = log productivity shocks [log(θb),log(θw)]
    #   πθ = transition matrix for θ
    #   Φ = basis matrix, ie, Φ*f = f(a,θ), for all a,θ
    #   EΦeg = expectation on gridpoints next period, eg EΦeg*fcoefs 
    #        = ∑ πθ(s,s′) ∑ fcoeff^j ϕ^j(a′,s′) 
    #   EΦ_aeg = derivative of the expectation with respect to assets
    #

    agrid::Vector{Float64} = zeros(Na)  
    abasis::Basis{1,Tuple{SplineParams{Vector{Float64}}}} =Basis(SplineParams(collect(LinRange(0,1,Na-1)),0,2))
    πθ::Matrix{Float64} =zeros(Nθ,Nθ)   
    lθ::Matrix{Float64} = zeros(Nθ,2)    
    Φ::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na)
    EΦeg::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na)  
    EΦ_aeg::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na) 

    #
    # Equilibrium results to be initialized
    #   alθ = gridpoints for the stationary distribution
    #   ω = stationary distribution 
    #   Λ = transition matrix for the stationary distribution
    #   r = interest rate
    #   w = wage rate
    #   tr = government transfers
    #   Vcoefs = coefficients of value function
    #

    alθ::Matrix{Float64} = zeros(Nθ*Ia,3) 
    ω::Vector{Float64} = zeros(Nh)   
    Λ::SparseMatrixCSC{Float64,Int64} = spzeros(Ia,Ia)
    r::Float64  = 0.0  
    w::Float64  = 0.0  
    tr::Float64 = 0.0  
    Vcoefs::Vector{Float64} = zeros(Nv) 
    wf::NamedTuple = (c=Vector{Spline1D}(), a=Vector{Spline1D}(), v=Vector{Spline1D}())
    bf::NamedTuple = (c=Vector{Spline1D}(), a=Vector{Spline1D}(), v=Vector{Spline1D}(),
                      k=Vector{Spline1D}(), n=Vector{Spline1D}(), y=Vector{Spline1D}(), π=Vector{Spline1D}())
    egi::Vector{Spline1D}=Vector{Spline1D}(undef,Nθ)
    ab_col_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds
    ab_bor_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds
    aw_bor_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds


end


"""

setup!(OCM::OCModel)

Setup grids and equilibrium guesses
Inputs: parameters (OCM)
Outputs: agrid,alθ,πθ,lθ,Φ,EΦeg,EΦ_aeg,ω,Λ,r,w,tr,Vcoefs

"""
function setup!(OCM::OCModel)

    @unpack a̲,Na,N_θb,N_θw,ρ_θw,σ_θw,bm_θw,ρ_θb,σ_θb,bm_θb,Nθ,
            βo,γ,σ,Ia,amax,so,curv_a,curv_h,trlb,trub,rlb,rub = OCM

    #Productivity shocks of workers
    if bm_θw==0
        mc = rouwenhorst(N_θw,ρ_θw,σ_θw)
        πθw = mc.p
        θwgrid = exp.(mc.state_values)
    else
        mc = tauchen(5,.70446,.1598256,0,3)
        πθw = mc.p
        θwgrid = exp.(mc.state_values)
    end
    
    #Productivity shocks of entrepreneurs
    if bm_θb==0
        mc = rouwenhorst(N_θb,ρ_θb,σ_θb)
        πθb = mc.p
        θbgrid = exp.(mc.state_values)
    else
        πθb = [0.611519   0.170401    0.0983162  0.0645004  0.0552636;
               0.172256   0.550903    0.187292   0.0643231  0.0252256;
               0.0986772  0.19074     0.475423   0.190404   0.0447556;
               0.0599203  0.0546977   0.1637     0.558095   0.163587;
               0.0455165  0.00943903  0.0342529  0.135187   0.775604]
        θbgrid = [0.4094057675495106,
                  0.6231521971523873,
                  0.948493381371056,
                  1.4436917636105182,
                  2.1974279939666546]
        θbgrid = θbgrid./θbgrid[3]
    end

    #Combined processes in one Markov chain
    θ  = [kron(ones(length(θwgrid)),θbgrid) kron(θwgrid,ones(length(θbgrid)))]
    πθ = OCM.πθ = kron(πθb,πθw)
    lθ = OCM.lθ = log.(θ)

    #Grid points and basis matrices for policy functions
    xvec = LinRange(0,1,Na-1).^curv_a 
    gridknots = a̲ .+ (amax - a̲).*xvec 
    abasis = OCM.abasis = Basis(SplineParams(gridknots,0,so))
    agrid = OCM.agrid = nodes(abasis)[1]
    Φ = OCM.Φ = kron(Matrix(I,Nθ,Nθ),BasisMatrix(abasis,Direct()).vals[1])
    OCM.EΦeg = kron(πθ,BasisMatrix(abasis,Direct(),agrid).vals[1])
    OCM.EΦ_aeg = kron(πθ,BasisMatrix(abasis,Direct(),agrid,[1]).vals[1])

    #Grid for histogram
    xvec    = LinRange(0,1,Ia).^curv_h
    hagrid  = a̲ .+ (amax - a̲).*xvec 
    OCM.alθ = hcat(kron(ones(Nθ),hagrid),kron(lθ,ones(Ia)))
    OCM.ω   = ones(2*Ia*Nθ)/(2*Ia*Nθ) 

    #Guess prices and transfers
    OCM.r   = 0.5*(rlb+rub)
    OCM.tr  = 0.5*(trlb+trub)
    OCM.w   = w = 1.4110781255203524
    OCM.tx  = 0.12728259420181093   

    #Ensure discount factor is updated properly
    OCM.β   = β = βo*(1+γ)^(-σ)

    #Guess for unknown coefficients
    c_guess = (1-β)*agrid .+ w
    OCM.Vcoefs  = Φ\repeat(c_guess.^(1-σ)./(1-σ)/(1-β),Nθ)

end


"""

p = probw(vb-vw,σ_ε)

Logistic function for taste shocks: 
Inputs: value differences (vb-vw), taste shock standard deviation (σ_ε)
Output: probability of being a worker (p)

"""
function probw(vblessvw::Float64,σ_ε::Float64)
  p = 1/(1+exp(vblessvw/σ_ε))
  return p
end


"""

cf,af = policyw(OCM)

EG method applied to worker dynamic program
Inputs: parameters (OCM)
Outputs: policy rules (cf,af)

"""
function policyw(OCM::OCModel)

    @unpack Vcoefs,σ,β,γ,Nθ,lθ,a̲,EΦ_aeg,agrid,r,w,tr,τc,τw = OCM

    lθw  = lθ[:,2]
    θw   = exp.(lθw)
    w̄    = (1-τw)*w

    #Compute value function derivative
    EVₐ′ = reshape(EΦ_aeg*Vcoefs,:,Nθ) 
    EVₐ′ = max.(EVₐ′,1e-6)

    #Compute consumption today implied by Euler equation
    cEE = (β.*EVₐ′).^(-1/σ) #consumption today

    #Compute asset today implied by savings and consumtion
    Implieda = ((1+γ).*agrid .+ (1+τc).*cEE .- w̄.*θw' .-tr)./(1+r)  

    #Initialize policy rules for each productivity
    cf = Vector{Spline1D}(undef,Nθ) 
    af = Vector{Spline1D}(undef,Nθ)

    #
    #Deal with borrowing constraints:
    #
    #   -- For all a∈[a̲,Implied_a(a′=̲a)] borrowing constraint is binding
    #   -- Check a[1,s]: 
    #      -- if bigger than lower bound, consumption from BC
    #      -- for a≤a[1,s], interpolate with cEE
    #      
    for s in 1:Nθ 
        #With some productivities the borrowing constraint does not bind
        if issorted(Implieda[:,s])
            if Implieda[1,s] > a̲ #borrowing constraint binds
                #add extra points on the borrowing constraint for interpolation
                â = [a̲;Implieda[:,s]]
                ĉ = [((1+r)*a̲-(1+γ)*a̲+w̄*θw[s]+tr)/(1+τc);cEE[:,s]]
                cf[s] = Spline1D(â,ĉ,k=1)
                af[s] = Spline1D(â,[a̲;agrid],k=1)
            else
                cf[s] = Spline1D(Implieda[:,s],cEE[:,s],k=1)
                af[s] = Spline1D(Implieda[:,s],agrid,k=1)
            end
        else
            p = sortperm(Implieda[:,s])
            if Implieda[p[1],s] > a̲ #borrowing constraint binds
                #Add extra points on the borrowing constraint for interpolation
                â = [a̲;Implieda[p,s]]
                ĉ = [((1+r)*a̲-(1+γ)a̲+w̄*θw[s]+tr)/(1+τc);cEE[p,s]]
                cf[s] = Spline1D(â,ĉ,k=1)
                af[s] = Spline1D(â,[a̲;agrid[p]],k=1)
            else
                cf[s] = Spline1D(Implieda[p,s],cEE[p,s],k=1)
                af[s] = Spline1D(Implieda[p,s],agrid[p],k=1)
            end
        end
    end
    return cf,af
end


"""

cf,af,kf,nf,yf,πf = policyb(OCM)

EG method applied to business dynamic program
Inputs: parameters (OCM), V coefficients (Vcoefs)
Outputs: policy rules (cf,af,kf,nf,yf,πf,ζf)

"""
function policyb(OCM::OCModel)
    @unpack Vcoefs,σ,β,γ,Nθ,lθ,a̲,EΦ_aeg,Na,agrid,α_b,ν,δ,χ,r,w,tr,τc,τb = OCM

    lθb = lθ[:,1]
    θb  = exp.(lθb)

    #Initialize policy rules for each productivity
    cf  = Vector{Spline1D}(undef,Nθ)
    af  = Vector{Spline1D}(undef,Nθ)
    kf  = Vector{Spline1D}(undef,Nθ)
    nf  = Vector{Spline1D}(undef,Nθ)
    yf  = Vector{Spline1D}(undef,Nθ)
    πf  = Vector{Spline1D}(undef,Nθ)

    #Compute firms profit (ignoring constraints)
    nbyk = ν*(r+δ)/(α_b*w) 
    kvec = @. (w/(ν*θb*nbyk^(ν-1)))^(1/(α_b+ν-1))
    πu   = @. θb*kvec^α_b*(nbyk*kvec)^ν-(r+δ)*kvec-w*(nbyk*kvec)

    #Compute value function derivative
    EVₐ′ = reshape(EΦ_aeg*Vcoefs,:,Nθ) 
    EVₐ′ = max.(EVₐ′,1e-6)

    #Compute consumption today implied by Euler equation
    cEE = (β.*EVₐ′).^(-1/σ) 

    #Compute asset today implied by savings and consumtion
    Implieda = ((1+γ).*agrid .+ (1+τc).*cEE .- (1-τb).*πu' .-tr) ./ (1+r) 

    #Find out where borrowing constraints bind
    k   = ones(Na).*kvec'
    y   = ones(Na).*(θb.*kvec.^α_b.*(nbyk.*kvec).^ν)'
    πb  = ones(Na).*πu'
    n   = nbyk.*k

    #Compute the argument for EGM Inverse 
    argEGMinv = (1+γ).*agrid .+  (1+τc).*cEE .- tr
    for s in 1:Nθ
        ic   = χ.*Implieda[:,s] .< kvec[s] 
        if sum(ic) > 0
            Implieda[ic,s] = OCM.egi[s](argEGMinv[ic,s])
            k[ic,s]  = max.(χ.*Implieda[ic,s],0)
            n[ic,s]  = (w./(ν.*θb[s].*k[ic,s].^α_b)).^(1/(ν-1))
            y[ic,s]  = θb[s].*k[ic,s].^α_b.*n[ic,s].^ν
            πb[ic,s] = y[ic,s] - w.*n[ic,s] - (r+δ).*k[ic,s]
        end
    end

    #Update where borrowing constraint binding and interpolate
    numa = 10
    for s in 1:Nθ
        min_a=minimum(Implieda[:,s])
        if min_a > a̲ 
            acon  = LinRange(a̲,min_a,numa+1)[1:end-1]
            kcon  = min.(χ.*acon,kvec[s])
            ncon  = (w./(ν.*θb[s].*kcon.^α_b)).^(1/(ν-1))
            ycon  = θb[s].*kcon.^α_b.*ncon.^ν
            πcon  = ycon - w.*ncon - (r+δ).*kcon
            ancon = a̲.*ones(numa)
            ccon  = ((1+r).*acon .+ (1-τb).*πcon .+ tr .- (1+γ)*a̲) ./(1+τc)
        end

        if issorted(Implieda[:,s])
            if Implieda[1,s] > a̲ #borrowing constraint binds
                #now interpolate
                cf[s] = Spline1D([acon;Implieda[:,s]],[ccon;cEE[:,s]],k=1)
                af[s] = Spline1D([acon;Implieda[:,s]],[ancon;agrid],k=1)
                kf[s] = Spline1D([acon;Implieda[:,s]],[kcon;k[:,s]],k=1)
                nf[s] = Spline1D([acon;Implieda[:,s]],[ncon;n[:,s]],k=1)
                yf[s] = Spline1D([acon;Implieda[:,s]],[ycon;y[:,s]],k=1)
                πf[s] = Spline1D([acon;Implieda[:,s]],[πcon;πb[:,s]],k=1)
            else
                cf[s] = Spline1D(Implieda[:,s],cEE[:,s],k=1)
                af[s] = Spline1D(Implieda[:,s],agrid,k=1)
                kf[s] = Spline1D(Implieda[:,s],k[:,s],k=1)
                nf[s] = Spline1D(Implieda[:,s],n[:,s],k=1)
                yf[s] = Spline1D(Implieda[:,s],y[:,s],k=1)
                πf[s] = Spline1D(Implieda[:,s],πb[:,s],k=1)
            end
        else
            p = sortperm(Implieda[:,s])
            if Implieda[p[1],s] > a̲ #borrowing constraint binds
                #now interpolate
                cf[s] = Spline1D([acon;Implieda[p,s]],[ccon;cEE[p,s]],k=1)
                af[s] = Spline1D([acon;Implieda[p,s]],[ancon;agrid[p]],k=1)
                kf[s] = Spline1D([acon;Implieda[p,s]],[kcon;k[p,s]],k=1)
                nf[s] = Spline1D([acon;Implieda[p,s]],[ncon;n[p,s]],k=1)
                yf[s] = Spline1D([acon;Implieda[p,s]],[ycon;y[p,s]],k=1)
                πf[s] = Spline1D([acon;Implieda[p,s]],[πcon;πb[p,s]],k=1)
            else
                cf[s] = Spline1D(Implieda[p,s],cEE[p,s],k=1)
                af[s] = Spline1D(Implieda[p,s],agrid[p],k=1)
                kf[s] = Spline1D(Implieda[p,s],k[p,s],k=1)
                nf[s] = Spline1D(Implieda[p,s],n[p,s],k=1)
                yf[s] = Spline1D(Implieda[p,s],y[p,s],k=1)
                πf[s] = Spline1D(Implieda[p,s],πb[p,s],k=1)
            end
        end

    end
    return cf,af,kf,nf,yf,πf
end

"""
egi[s] = setup_egi!(OCM)

Computes the spline inverse in the EG algorithm
Inputs: parameters (OCM)
Outputs: eg inverse for each shock s
"""
function setup_egi!(OCM::OCModel)

    @unpack Nθ,lθ,α_b,ν,χ,r,w,δ,τb = OCM

    #Compute unconstrained capital
    lθb  = lθ[:,1]
    θb   = exp.(lθb)
    nbyk = ν*(r+δ)/(α_b*w) 
    kvec = @. (w/(ν*θb*nbyk^(ν-1)))^(1/(α_b+ν-1))

    #Compute EG inverse spline
    numk = 10
    for s in 1:Nθ
        ahold    = LinRange(-5.,kvec[s]/χ,numk) 
        khold    = max.(ahold.*χ,0)
        nhold    = (w./(ν.*θb[s].*khold.^α_b)).^(1/(ν-1))
        yhold    = θb[s].*khold.^α_b.*nhold.^ν
        πbhold   = yhold - w.*nhold - (r+δ).*khold
        aret     = (1+r).*ahold .+ (1 .- τb).*πbhold
        OCM.egi[s] = Spline1D(aret,ahold,k=1,bc="extrapolate")
    end
end 



"""
V,wf,bf = solve_eg!(OCM)

Solves the functional equation for value function V
Inputs: parameters (OCM)
Outputs: V coefficients (Vcoefs), wf,bf (policies)

"""
function solve_eg!(OCM::OCModel)
 
    @unpack Na,agrid,abasis,σ,β,Φ,Nθ,lθ,πθ,σ_ε,λ,Nit,tolegm = OCM

    Vhold = OCM.Vcoefs

    # Placeholders for splines
    cf_w  = af_w = Vf_w = nothing
    cf_b  = af_b = kf = nf = yf = πf = Vf_b = nothing

    #Iterate on the value function coefficients
    diff  = 1.
    dchg  = 1.
    tol   = tolegm
    luΦ   = lu(Φ)
    it    = 0
    while diff > tol && it < Nit && dchg > 1e-5

      #Compute optimal consumption, asset, and value functions
      cf_w,af_w = policyw(OCM)
      cf_b,af_b,kf,nf,yf,πf = policyb(OCM)

      #Compute values at gridpoints
      c_w,c_b = zeros(Na,Nθ),zeros(Na,Nθ) 
      a_w,a_b = zeros(Na,Nθ),zeros(Na,Nθ)
      Vw,Vb   = zeros(Na,Nθ),zeros(Na,Nθ)
      for s in 1:Nθ
          c_w[:,s] = cf_w[s](agrid) 
          c_b[:,s] = cf_b[s](agrid) 
          a_w[:,s] = af_w[s](agrid) 
          a_b[:,s] = af_b[s](agrid) 
          EΦw      = kron(πθ[s,:]',BasisMatrix(abasis,Direct(),a_w[:,s]).vals[1])
          EΦb      = kron(πθ[s,:]',BasisMatrix(abasis,Direct(),a_b[:,s]).vals[1])
          Vw[:,s]  = c_w[:,s].^(1-σ)/(1-σ) + β.*EΦw*OCM.Vcoefs
          Vb[:,s]  = c_b[:,s].^(1-σ)/(1-σ) + β.*EΦb*OCM.Vcoefs 
      end
      p       = probw.(Vb.-Vw,σ_ε)
      V       = p.*Vw .+ (1 .- p).*Vb
      ptol    = 1e-8
      ip      = ptol.< p .< 1-ptol
      V[ip]  .= Vw[ip] .+ σ_ε.*log.(1 .+ exp.((Vb[ip].-Vw[ip])./σ_ε))

      #Implied value functions
      Vf_w    = [Spline1D(agrid,Vw[:,s],k=1) for s in 1:Nθ]
      Vf_b    = [Spline1D(agrid,Vb[:,s],k=1) for s in 1:Nθ]

      #Update the coefficients using the linear system ΦVcoefs = V
      Vcnew   = luΦ\V[:]
      diff    = norm(OCM.Vcoefs.-Vcnew)
      OCM.Vcoefs = λ .* Vcnew + (1-λ) .* Vhold

      it     += 1
    end

    OCM.wf  = (c=cf_w,a=af_w,v=Vf_w)
    OCM.bf  = (c=cf_b,a=af_b,v=Vf_b,k=kf,n=nf,y=yf,π=πf)

    if it>= Nit
        println("solve_eg did not converge: $diff")
    else
        println("solve_eg converged in $it iterations: $diff")
    end
    if dchg<=1e-5
        println("solve_eg is making no progress: $dchg")
    end

end

"""

cdst,adst,vdst,ybdst,kbdst,nbdst,nwdst = dist!(OCM::OCModel)

Computes the stationary distribution 
Inputs: parameters (OCM)
Outputs: consumption, asset, value, capital, and labor distributions

"""
function dist!(OCM::OCModel)

    @unpack Vcoefs,wf,bf,Nθ,lθ,πθ,Ia,alθ,r,w,σ_ε = OCM

    ah  = alθ[1:Ia,1] #grids are all the same for all shocks
    afw = max.(min.(hcat([wf.a[s](ah) for s in 1:Nθ]...),ah[end]),ah[1]) 
    afb = max.(min.(hcat([bf.a[s](ah) for s in 1:Nθ]...),ah[end]),ah[1]) 
    Vw  = hcat([wf.v[s](ah) for s in 1:Nθ]...)
    Vb  = hcat([bf.v[s](ah) for s in 1:Nθ]...)
    p = probw.(Vb.-Vw,σ_ε)
    
    Qsw = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ah,0,1)),Direct(),@view afw[:,s]).vals[1]') for s in 1:Nθ]
    Qsb = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ah,0,1)),Direct(),@view afb[:,s]).vals[1]') for s in 1:Nθ]
    Λtemp = hcat(Qsw...,Qsb...)
    OCM.Λ = vcat(p[:].*Λtemp,(1 .- p[:]).*Λtemp)

    OCM.ω .=  real(eigsolve(OCM.Λ,OCM.ω,1)[2])[1]
    OCM.ω ./= sum(OCM.ω)
    nb     = hcat([bf.n[s](ah) for s in 1:Nθ]...)
    kb     = hcat([bf.k[s](ah) for s in 1:Nθ]...)
    yb     = hcat([bf.y[s](ah) for s in 1:Nθ]...)
    vw     = hcat([wf.v[s](ah) for s in 1:Nθ]...)
    vb     = hcat([bf.v[s](ah) for s in 1:Nθ]...)
    cw     = hcat([wf.c[s](ah) for s in 1:Nθ]...)
    cb     = hcat([bf.c[s](ah) for s in 1:Nθ]...)
    nwdst  = [exp.(alθ[:,3]);zeros(Ia*Nθ)]
    nbdst  = [zeros(Ia*Nθ);nb[:]]
    kbdst  = [zeros(Ia*Nθ);kb[:]]
    ybdst  = [zeros(Ia*Nθ);yb[:]]
    vdst   = [vw[:];vb[:]]
    cdst   = [cw[:];cb[:]]
    adst   = hcat([alθ[:,1];alθ[:,1]])
    
    return cdst,adst,vdst,ybdst,kbdst,nbdst,nwdst
end



"""

x = newton fixed point of f(x;ξ)=0

Simple newton
Inputs: initial guess (x0), relaxation (ξ)
Output: fixed point (x)

"""
function newton(f::Function, x0::Vector{Float64}, ξ::Float64;
                tol::Float64 = 1e-8, maxit::Int = 20)

    x   = copy(x0)                     # preserve original input
    lx  = length(x)
    fx  = similar(x)                   # f(x)
    fxp = similar(x)                   # f(xp)
    dx  = similar(x)                   # Newton step
    J   = zeros(Float64, lx, lx)       # Jacobian

    for it in 1:maxit
        fx .= f(x)                     # res = f(x)
        if norm(fx) < tol
            return x
        end

        @views begin
            del = 1e-2 .* abs.(x) .+ 1e-8

            for i in 1:lx
                xi = x[i]
                x[i] = xi + del[i]
                fxp .= f(x)
                x[i] = xi             # restore x[i]
                J[:, i] .= (fxp .- fx) ./ del[i]
            end

            dx .= J \ fx              # solve J dx = fx
            x  .-= ξ .* dx            # update step
        end
    end

    println("      NEWTON WARNING:")
    @printf("      Increase maxit %9.0f\n", maxit)
    return x
end
    

"""

ss  = solvess!(OCM::OCModel)

Solves for the steady state without aggregate shocks
Inputs: parameters (OCM)
Outputs: steady state values of interest (ss)

"""
function solvess!(OCM::OCModel)
    @unpack Θ̄,α,δ,γ,g,b,τc,τd,τp,τw,τb,trlb,trub,rlb,rub,Neval,iagg,ibise,iprint,ftolmk,xtolxmk,ξ,maxitermk,inewt = OCM

    ss      = zeros(5)
    lev     = zeros(32)
    shr     = zeros(32)

    x0      = zeros(2)
    xp      = zeros(2)
    fn      = zeros(2)
    fnp     = zeros(2)
    jac     = zeros(2,2)
    del     = zeros(2,2)


    function ss1Res(x)
        OCM.r  = r = x[1]
        rc     = r/(1-τp)
        K2Nc   = ((rc+δ)/(Θ̄*α))^(1/(α-1))
        OCM.w  = w = (1-α)*Θ̄*K2Nc^α

        setup_egi!(OCM)
        solve_eg!(OCM)
        cdst,adst,vdst,ybdst,kbdst,nbdst,nwdst = dist!(OCM)
        A      = dot(OCM.ω,adst)
        Nb     = dot(OCM.ω,nbdst)
        Nc     = dot(OCM.ω,nwdst)-Nb
        Kc     = K2Nc*Nc
        Kb     = dot(OCM.ω,kbdst)
        res    = 1-((1-τd)*Kc+Kb+b)/A

        if iprint==1
            @printf("      Interest rate %10.2f\n",r*100)
            @printf("      Asset market  %10.3e\n",res)
            println("")
        end
        return res
    end


    function ssRes(x)

        OCM.r  = r = x[1]
        OCM.tr = tr = x[2]
        rc     = r/(1-τp)
        K2Nc   = ((rc+δ)/(Θ̄*α))^(1/(α-1))
        OCM.w  = w = (1-α)*Θ̄*K2Nc^α

        setup_egi!(OCM)
        solve_eg!(OCM)
        cdst,adst,vdst,ybdst,kbdst,nbdst,nwdst = dist!(OCM)

        A      = dot(OCM.ω,adst)
        Nb     = dot(OCM.ω,nbdst)
        Nc     = dot(OCM.ω,nwdst)-Nb
        Kc     = K2Nc*Nc
        Yc     = Θ̄*Kc^α*Nc^(1-α)
        Kb     = dot(OCM.ω,kbdst)
        Yb     = dot(OCM.ω,ybdst)
        C      = dot(OCM.ω,cdst)
        Tc     = τc*C
        Tp     = τp*(Yc-w*Nc-δ*Kc)
        Td     = τd*(Yc-w*Nc-(γ+δ)*Kc-Tp)
        Tw     = τw*w*(Nc+Nb)
        Tb     = τb*(Yb-(r+δ)*Kb-w*Nb)
        OCM.tx = Tc+Tp+Td+Tw+Tb
        res    = [1-((1-τd)*Kc+Kb+b)/A,1-(OCM.tx-tr-(OCM.r-γ)*b)/g]

        if iprint==1
            @printf("      Tax on business %10.2f, Tax on wages %10.2f\n",τb*100,τw*100)
            @printf("      Interest rate %10.2f, Govt transfer %10.2f\n",r*100,tr)
            @printf("      Asset market  %10.3e, Govt budget   %10.3e\n",res[1],res[2])
            println("")
        end
        return res[1:2]

    end

    if ibise==1
        @printf("      Using bisection method\n")
        OCM.tr  = trlb
        ret1    = find_zero(ss1Res,(rlb,rub),Bisection(),maxevals=Neval,atol=1e-8)
        res1    = ss1Res(ret1)
        OCM.tr  = trub
        ret2    = find_zero(ss1Res,(rlb,rub),Bisection(),maxevals=Neval,atol=1e-8)
        res2    = ss1Res(ret2)
        if abs(res1)<abs(res2)
            OCM.r  = ret1 
            OCM.tr = trlb
        else
            OCM.r  = ret2
            OCM.tr = trub
        end
    else
        OCM.r  = 0.5*(rlb+rub)
        OCM.tr = 0.5*(trlb+trub)
    end

    if inewt==1
        println("      Using newton method")
        rtr    = newton(ssRes,[OCM.r,OCM.tr],ξ,tol=ftolmk)   
        OCM.r  = rtr[1]
        OCM.tr = rtr[2]
        res    = ssRes(rtr) 
    else
        ret    = nlsolve(ssRes,[OCM.r,OCM.tr],ftol=1e-6)
        OCM.r  = ret.zero[1]
        OCM.tr = ret.zero[2]
        res    = ssRes(ret.zero)
    end

    if iprint==0
        @printf("      Interest rate %10.2f, Govt transfer %10.2f\n",OCM.r*100,OCM.tr)
        @printf("      Asset market  %10.3e, Govt budget   %10.3e\n",res[1],res[2])
        println("")
    end

    bshr   = sum(reshape(OCM.ω,:,2),dims=1)[2]
    ss     = [OCM.r*100,OCM.tr,res[1],res[2],bshr*100] 
   
    if iagg==1
        rc    = OCM.r/(1-τp)
        K2Nc  = ((rc+δ)/(Θ̄*α))^(1/(α-1))
        w     = (1-α)*Θ̄*K2Nc^α

        cdst,adst,vdst,ybdst,kbdst,nbdst,nwdst = dist!(OCM)
        Nb    = dot(OCM.ω,nbdst)
        Nc    = dot(OCM.ω,nwdst)-Nb
        Kc    = K2Nc*Nc
        Yc    = Θ̄*Kc^α*Nc^(1-α)
        Kb    = dot(OCM.ω,kbdst)
        Yb    = dot(OCM.ω,ybdst)
        C     = dot(OCM.ω,cdst)
        Tc    = τc*C
        Tp    = τp*(Yc-w*Nc-δ*Kc)
        Td    = τd*(Yc-w*Nc-(γ+δ)*Kc-Tp)
        Tn    = τw*w*(Nc+Nb)
        Tb    = τb*(Yb-(OCM.r+δ)*Kb-w*Nb)
        Tx    = Tc+Tp+Td+Tn+Tb
        rc    = OCM.r/(1-τp)
        lev   = [C,(γ+δ)*Kb,(γ+δ)*Kc,g,Yc+Yb,
                    Yb-(OCM.r+δ)*Kb-w*Nb,w*(Nc+Nb),
                    Yc-w*Nc-δ*Kc,OCM.r*Kb,δ*(Kc+Kb),Yc+Yb,
                    Tb,Tn,Tp,Td,Tc,Tx,
                    g,(OCM.r-γ)*b,OCM.tr,g+(OCM.r-γ)*b+OCM.tr,
                    w*Nc,rc*Kc,δ*Kc,Yc,Kc,
                    w*Nb,OCM.r*Kb,δ*Kb,Yb-(OCM.r+δ)*Kb-w*Nb,Yb,Kb]
        den   = hcat((Yc+Yb) .*ones(1,21),
                      Yc .* ones(1,5),Yb .* ones(1,6))
        shr   = (lev[:] ./ den[:]) .* 100.0
    end
    return ss,lev,shr,res
end


"""

res = check!(OCM::OCModel)

Checks residuals varying the interest rate and transfers
Inputs: parameters (OCM)
Outputs: equilibrium residuals

"""
function check!(OCM::OCModel)
    @unpack Θ̄,α,δ,γ,g,b,τc,τd,τp,τw,τb,trlb,trub,Ntr,rlb,rub,Neval,Nr = OCM

    rvals   = LinRange(rlb,rub,Nr)
    trvals  = LinRange(trlb,trub,Ntr)
    chk1    = zeros(Ntr,Nr)
    chk2    = zeros(Ntr,Nr)

    function ssRes(x)

        OCM.r  = r = x[1]
        OCM.tr = tr = x[2]
        rc     = r/(1-τp)
        K2Nc   = ((rc+δ)/(Θ̄*α))^(1/(α-1))
        OCM.w  = w = (1-α)*Θ̄*K2Nc^α

        setup_egi!(OCM)
        solve_eg!(OCM)
        cdst,adst,vdst,ybdst,kbdst,nbdst,nwdst = dist!(OCM)

        Nb     = dot(OCM.ω,nbdst)
        Nc     = dot(OCM.ω,nwdst)-Nb
        Kc     = K2Nc*Nc
        Yc     = Θ̄*Kc^α*Nc^(1-α)
        Kb     = dot(OCM.ω,kbdst)
        Yb     = dot(OCM.ω,ybdst)
        C      = dot(OCM.ω,cdst)
        Tc     = τc*C
        Tp     = τp*(Yc-w*Nc-δ*Kc)
        Td     = τd*(Yc-w*Nc-(γ+δ)*Kc-Tp)
        Tn     = τw*(Nc+Nb)
        Tb     = τb*(Yb-(r+δ)*Kb-w*Nb)

        OCM.tx = Tc+Tp+Td+Tn+Tb

        kres   = dot(OCM.ω,adst)-(1-τd)*Kc-Kb-b
        gres   = g+(r-γ)*b+tr-OCM.tx

        return kres,gres

    end

    for j in 1:Nr
        for i in 1:Ntr
            kres,gres = ssRes([rvals[j],trvals[i]])
            chk1[i,j] = kres 
            chk2[i,j] = gres 
        end
    end

    return chk1,chk2
end


function assign!(OCM::OCModel,r::Float64,tr::Float64)

    setup!(OCM)
    OCM.r  = r
    OCM.tr = tr

    @unpack τp,τd,τb,τc,τw,δ,Θ̄,α,b,γ,g = OCM

    rc     = r/(1-τp)
    K2Nc   = ((rc+δ)/(Θ̄*α))^(1/(α-1))
    OCM.w  = w = (1-α)*Θ̄*K2Nc^α

    setup_egi!(OCM)
    solve_eg!(OCM)
    updatecutoffs!(OCM)
    cdst,adst,vdst,ybdst,kbdst,nbdst,nwdst = dist!(OCM)

    Nb     = dot(OCM.ω,nbdst)
    Nc     = dot(OCM.ω,nwdst)-Nb
    Kc     = K2Nc*Nc
    Yc     = Θ̄*Kc^α*Nc^(1-α)
    Kb     = dot(OCM.ω,kbdst)
    Yb     = dot(OCM.ω,ybdst)
    C      = dot(OCM.ω,cdst)
    Tc     = τc*C
    Tp     = τp*(Yc-w*Nc-δ*Kc)
    Td     = τd*(Yc-w*Nc-(γ+δ)*Kc-Tp)
    Tn     = τw*w*(Nc+Nb)
    Tb     = τb*(Yb-(r+δ)*Kb-w*Nb)
    OCM.tx = Tc+Tp+Td+Tn+Tb

    tem    = adst-((1-τd)*K2Nc) .* (nwdst-nbdst)-kbdst
    res    = [dot(OCM.ω,tem)-b,g+(OCM.r-γ)*b+OCM.tr-OCM.tx]
    @printf("  Asset market  %10.3e, Govt budget   %10.3e\n",res[1],res[2])
    diffv=diffegm(OCM)
    @printf("  Diff in EGM     %10.3e\n",diffv)

end


function updatecutoffs!(OCM::OCModel)

    @unpack alθ,Ia,Nθ,lθ, χ,agrid,  a̲, Na,Ia, ab_col_cutoff,ab_bor_cutoff, aw_bor_cutoff, bf,wf = OCM
    
    # where do constraints bind?
    ah  = agrid #grids are all the same for all shocks
    ah=alθ[1:Ia,1]
    kb     = hcat([bf.k[s](ah) for s in 1:Nθ]...)

    # for each shock, find the borrowing constraint
    for s in 1:Nθ
        indices = findall(kb[:,s] .≈ ah*χ)
        ab_col_cutoff[lθ[s,:]] = maximum(indices)==1 ?   -Inf : ah[maximum(indices)]
    end



    # borrowing constraint for owners
    a′ =hcat([bf.a[s](ah) for s in 1:Nθ]...)
    # for each shock, find the borrowing constraint
    for s in 1:Nθ
        indices = findall(a′[:,s] .> 0)
        ab_bor_cutoff[lθ[s,:]] = minimum(indices)==1 ?   -Inf : ah[minimum(indices)-1]
    end

    # borrowing constraint for workers
    a′ =hcat([wf.a[s](ah) for s in 1:Nθ]...)
    # for each shock, find the borrowing constraint
    for s in 1:Nθ
        indices = findall(a′[:,s] .> 0)
        aw_bor_cutoff[lθ[s,:]] = minimum(indices)==1 ?   -Inf : ah[minimum(indices)-1]
    end

end


 """
    save_policy_functions!(OCM::OCModel)

Saves the policy functions in the OCModel object
"""

function get_policy_functions(OCM::OCModel)
    @unpack bf,wf,curv_a,Na,amax,a̲,curv_h,Ia,r,σ,δ,α_b,agrid,Nθ,lθ,ν,χ=OCM



    
    λb = Vector{Spline1D}(undef,Nθ)
    λw = Vector{Spline1D}(undef,Nθ)
    cb=zeros(Na)
    cw=zeros(Na)
    kb=zeros(Na)
    nb=zeros(Na)
    mpk=zeros(Na)
    ζval=zeros(Na)
    for s in 1:Nθ
        cb .= bf.c[s](agrid)
        cw .= wf.c[s](agrid)
        kb .= max.(bf.k[s](agrid),1e-4)
        nb .= bf.n[s](agrid)
        z   = exp.(lθ[s,1]) # productivity shock
        mpk .= α_b*z*kb.^(α_b-1).*nb.^ν
        ζval .= cb.^(-σ).*(mpk .-r .-δ)
        λb[s]  = Spline1D(agrid,(1+r)*cb.^(-σ)+χ*ζval,k=1)   
        λw[s]  = Spline1D(agrid,(1+r)*cw.^(-σ),k=1)
    end


    #save the policy functions a,n,k,λ,v
    af(lθ,a,c) = c==1 ? wf.a[[lθ].==eachrow(OCM.lθ)][1](a) : bf.a[[lθ].==eachrow(OCM.lθ)][1](a)
    nf(lθ,a,c) = c==1 ? -exp.(lθ[2]) : bf.n[[lθ].==eachrow(OCM.lθ)][1](a)
    kf(lθ,a,c) = c==1 ? 0 : bf.k[[lθ].==eachrow(OCM.lθ)][1](a)
    yf(lθ,a,c) = c==1 ? 0 : bf.y[[lθ].==eachrow(OCM.lθ)][1](a)
    nbf(lθ,a,c) = c==1 ? 0 : bf.n[[lθ].==eachrow(OCM.lθ)][1](a)
    cf(lθ,a,c) = c==1 ? wf.c[[lθ].==eachrow(OCM.lθ)][1](a) : bf.c[[lθ].==eachrow(OCM.lθ)][1](a)
    λf(lθ,a,c) = c==1 ? λw[[lθ].==eachrow(OCM.lθ)][1](a) : λb[[lθ].==eachrow(OCM.lθ)][1](a)
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

    #next get kinks
    ℵ = Int[]
    #for s in 1:nθ
    #    if OCM.a_cutoff[θ[s]] > -Inf
    #        push!(ℵ,findlast(a_sp .< OCM.a_cutoff[θ[s]])+(s-1)*nsp)
    #    end
    #end 
    mask = OCM.ω .> 1e-10
    println("Maximum assets: $(maximum(aθc_Ω[mask,1]))")

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



function diffegm(OCM::OCModel) 
        @unpack Na,agrid,abasis,σ,β,Φ,Nθ,lθ,πθ,σ_ε,λ,Nit,tolegm = OCM

    Vhold = OCM.Vcoefs

    # Placeholders for splines
    cf_w  = af_w = Vf_w = nothing
    cf_b  = af_b = kf = nf = yf = πf = Vf_b = nothing

    #Iterate on the value function coefficients
    diff  = 1.
    dchg  = 1.
    tol   = tolegm
    luΦ   = lu(Φ)
    it    = 0
      #Compute optimal consumption, asset, and value functions
      cf_w,af_w = policyw(OCM)
      cf_b,af_b,kf,nf,yf,πf = policyb(OCM)

      #Compute values at gridpoints
      c_w,c_b = zeros(Na,Nθ),zeros(Na,Nθ) 
      a_w,a_b = zeros(Na,Nθ),zeros(Na,Nθ)
      Vw,Vb   = zeros(Na,Nθ),zeros(Na,Nθ)
      for s in 1:Nθ
          c_w[:,s] = cf_w[s](agrid) 
          c_b[:,s] = cf_b[s](agrid) 
          a_w[:,s] = af_w[s](agrid) 
          a_b[:,s] = af_b[s](agrid) 
          EΦw      = kron(πθ[s,:]',BasisMatrix(abasis,Direct(),a_w[:,s]).vals[1])
          EΦb      = kron(πθ[s,:]',BasisMatrix(abasis,Direct(),a_b[:,s]).vals[1])
          Vw[:,s]  = c_w[:,s].^(1-σ)/(1-σ) + β.*EΦw*OCM.Vcoefs
          Vb[:,s]  = c_b[:,s].^(1-σ)/(1-σ) + β.*EΦb*OCM.Vcoefs
      end
      p       = probw.(Vb.-Vw,σ_ε)
      V       = p.*Vw .+ (1 .- p).*Vb
      ptol    = 1e-8
      ip      = ptol.< p .< 1-ptol
      V[ip]  .= Vw[ip] .+ σ_ε.*log.(1 .+ exp.((Vb[ip].-Vw[ip])./σ_ε))

      #Implied value functions
      Vf_w    = [Spline1D(agrid,Vw[:,s],k=1) for s in 1:Nθ]
      Vf_b    = [Spline1D(agrid,Vb[:,s],k=1) for s in 1:Nθ]

      #Update the coefficients using the linear system ΦVcoefs = V
      Vcnew   = luΦ\V[:]
      diff    = norm(OCM.Vcoefs.-Vcnew)
      return diff

end


function solvecase(τb, τw, r_guess, tr_guess)
    diff_v = Inf
    res = zeros(2)
    try
        # Recreate a fresh copy of OCM for each worker
        OCM = OCModel()
        OCM.τb = τb
        OCM.τw = τw

        assign!(OCM, r_guess, tr_guess)
        OCM.ibise = 1
        ss, lev, shr, res = solvess!(OCM)
        diff_v = diffegm(OCM)
        Xss = getX(OCM)  # [R, W, Tr, Frac_b, V, A, C]
        R, W, Tr, Frac_b, V, A, C = Xss

        return NamedTuple{
            (:τb, :τw, :r, :tr, :diffv, :diffasset, :diffgbc, :Rss, :Wss, :Trss, :Frac_bss, :Vss, :Ass, :Css)
        }((τb, τw, OCM.r, OCM.tr, diff_v, res[1], res[2], R, W, Tr, Frac_b, V, A, C))

    catch e
        @warn "Solver failed at τb = $τb, τw = $τw" exception=(e, catch_backtrace())
        return NamedTuple{
            (:τb, :τw, :r, :tr, :diffv, :diffasset, :diffgbc, :Rss, :Wss, :Trss, :Frac_bss, :Vss, :Ass, :Css)
        }((τb, τw, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN))
    end
end

function guess_from_csv(τb, τw, df)
    row = df[(df.τb .≈ τb) .& (df.τw .≈ τw), :]
    if nrow(row) == 1 && !isnan(row.r[1]) && !isnan(row.tr[1])
        return row.r[1], row.tr[1]
    else
        # Prepare data: each column is a point in 2D
        pts = permutedims(hcat(df.τb, df.τw))  # 2×N matrix
        query = reshape([τb, τw], 2, 1)        # 2×1 matrix
        dists = pairwise(Euclidean(), pts, query)  # (N×1 matrix)
        idx = argmin(dists)
        return df.r[idx], df.tr[idx]
    end
end





#
# Functions for nonlinear transition path
#

function forward!(OCM,ω,ωn,r,tr,wf,bf)

    @unpack Nθ,πθ,lθ,Ia,alθ,δ,Θ̄,α,τp,τc,τb,τw,γ,σ_ε = OCM

    K2Nc   = ((r/(1-τp)+δ)/(Θ̄*α))^(1/(α-1))
    w      = (1-α)*Θ̄*K2Nc^α

    ah     = alθ[1:Ia,1] 
    cw     = hcat([wf.c[s](ah) for s in 1:Nθ]...)
    cb     = hcat([bf.c[s](ah) for s in 1:Nθ]...)
    πb     = hcat([bf.π[s](ah) for s in 1:Nθ]...)
    Vw     = hcat([wf.v[s](ah) for s in 1:Nθ]...)
    Vb     = hcat([bf.v[s](ah) for s in 1:Nθ]...)
    p      = probw.(Vb.-Vw,σ_ε)

    anw    = ((1+r) .* ah .+ (1-τw)*w.*exp.(lθ[:,2]') .- (1+τc) .*cw .+tr) ./(1+γ)
    anb    = ((1+r) .* ah .+ (1-τb).*πb .- (1+τc) .*cb .+tr) ./(1+γ)
    anw    = max.(min.(anw,ah[end]),ah[1])
    anb    = max.(min.(anb,ah[end]),ah[1])

    # Qsw    = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ah,0,1)),Direct(),@view anw[:,s]).vals[1]') for s in 1:Nθ]
    # Qsb    = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ah,0,1)),Direct(),@view anb[:,s]).vals[1]') for s in 1:Nθ]
    # Λtemp  = hcat(Qsw...,Qsb...)
    # Λ      = vcat(p[:].*Λtemp,(1 .- p[:]).*Λtemp)

        # Precompute spline basis only once
    B = Basis(SplineParams(ah, 0, 1))

    # Precompute sparse Qsw and Qsb matrices
    Qsw = [sparse(kron(πθ[s, :], BasisMatrix(B, Direct(), @view anw[:, s]).vals[1]')) for s in 1:Nθ]
    Qsb = [sparse(kron(πθ[s, :], BasisMatrix(B, Direct(), @view anb[:, s]).vals[1]')) for s in 1:Nθ]

    # Build Λtemp = hcat(Qsw..., Qsb...) (size: 25_000 × 50_000)
    Λtemp = hcat(Qsw..., Qsb...)
    vbuf= zeros(Ia * Nθ )  # Buffer for the result
    # Sparse matvec multiply without building full Λ
    #Λtempω = Λtemp * ω
    update_density!(ωn, Λtemp, p[:], ω,Ia,Nθ,vbuf)  # **fast call**
    # ωn    .= Λ*ω
     ωn   ./= sum(ωn)

end 



function forward_alt!(OCM,ω,ωn,r,tr,wf,bf)

    @unpack Nθ,πθ,lθ,Ia,alθ,δ,Θ̄,α,τp,τc,τb,τw,γ,σ_ε = OCM

    K2Nc   = ((r/(1-τp)+δ)/(Θ̄*α))^(1/(α-1))
    w      = (1-α)*Θ̄*K2Nc^α

    ah     = alθ[1:Ia,1] 
    cw     = hcat([wf.c[s](ah) for s in 1:Nθ]...)
    cb     = hcat([bf.c[s](ah) for s in 1:Nθ]...)
    πb     = hcat([bf.π[s](ah) for s in 1:Nθ]...)
    Vw     = hcat([wf.v[s](ah) for s in 1:Nθ]...)
    Vb     = hcat([bf.v[s](ah) for s in 1:Nθ]...)
    p      = probw.(Vb.-Vw,σ_ε)

    anw    = ((1+r) .* ah .+ (1-τw)*w.*exp.(lθ[:,2]') .- (1+τc) .*cw .+tr) ./(1+γ)
    anb    = ((1+r) .* ah .+ (1-τb).*πb .- (1+τc) .*cb .+tr) ./(1+γ)
    anw    = max.(min.(anw,ah[end]),ah[1])
    anb    = max.(min.(anb,ah[end]),ah[1])

    # Qsw    = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ah,0,1)),Direct(),@view anw[:,s]).vals[1]') for s in 1:Nθ]
    # Qsb    = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ah,0,1)),Direct(),@view anb[:,s]).vals[1]') for s in 1:Nθ]
    # Λtemp  = hcat(Qsw...,Qsb...)
    # Λ      = vcat(p[:].*Λtemp,(1 .- p[:]).*Λtemp)

        # Precompute spline basis only once
    B = Basis(SplineParams(ah, 0, 1))
    #compute transition matrix for exogenous states 
    Qθs = [kron(πθ[s,:], sparse(I, Ia, Ia)) for s in 1:Nθ]  # Sparse matrix for each shock state
    #next transition matrix for occupational choice 
    Pw = spdiagm(p[:])                          # Diagonal matrix for worker choice probabilities
    Pb = spdiagm(1.0 .- p[:])           # Diagonal matrix for business owner choice probabilities
    P = vcat(Pw, Pb)  # Combine into a block diagonal matrix

    ω = reshape(ω, Ia,Nθ, 2)  # Reshape ω to a matrix of size (Ia*Nθ, 2)
    ωn .= 0.0
    for s in 1:Nθ
        # Compute the transition probabilities for workers and business owners
        Qsw = BasisMatrix(B, Direct(), @view anw[:, s]).vals[1]'
        Qsb = BasisMatrix(B, Direct(), @view anb[:, s]).vals[1]'
        
        ωn .+= P*Qθs[s]*(Qsw * ω[:, s, 1] + Qsb * ω[:, s, 2])  # Apply the transition probabilities
    end
    ωn   ./= sum(ωn)

end 

function update_density!(ωn, Λtemp, p, ω,Ia, Nθ,v)
    v = Λtemp * ω                 # 25 000-vector
    @views begin
        ωn[1:Ia*Nθ]         .= p .* v
        ωn[Ia*Nθ+1:end]     .= (1 .- p) .* v
    end
    return nothing
end

function forward2!(OCM,ω,ωn,r,tr,wf,bf)

    @unpack Nθ,πθ,lθ,Ia,alθ,δ,Θ̄,α,τp,τc,τb,τw,γ,σ_ε = OCM

    K2Nc   = ((r/(1-τp)+δ)/(Θ̄*α))^(1/(α-1))
    w      = (1-α)*Θ̄*K2Nc^α

    ah     = alθ[1:Ia,1] 
    cw     = hcat([wf.c[s](ah) for s in 1:Nθ]...)
    cb     = hcat([bf.c[s](ah) for s in 1:Nθ]...)
    πb     = hcat([bf.π[s](ah) for s in 1:Nθ]...)
    Vw     = hcat([wf.v[s](ah) for s in 1:Nθ]...)
    Vb     = hcat([bf.v[s](ah) for s in 1:Nθ]...)
    p      = probw.(Vb.-Vw,σ_ε)

    anw    = ((1+r) .* ah .+ (1-τw)*w.*exp.(lθ[:,2]') .- (1+τc) .*cw .+tr) ./(1+γ)
    anb    = ((1+r) .* ah .+ (1-τb).*πb .- (1+τc) .*cb .+tr) ./(1+γ)
    anw    = max.(min.(anw,ah[end]),ah[1])
    anb    = max.(min.(anb,ah[end]),ah[1])

    Qsw    = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ah,0,1)),Direct(),@view anw[:,s]).vals[1]') for s in 1:Nθ]
    Qsb    = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ah,0,1)),Direct(),@view anb[:,s]).vals[1]') for s in 1:Nθ]
    Λtemp  = hcat(Qsw...,Qsb...)
    Λ      = vcat(p[:].*Λtemp,(1 .- p[:]).*Λtemp)
    ωn    .= Λ*ω
    ωn   ./= sum(ωn)

end 

    
function policy_path(OCM,rT,trT)

    @unpack Na,agrid,abasis,σ,β,Φ,Nθ,lθ,πθ,σ_ε,α,Θ̄,τp,δ = OCM

    T       = length(rT)
    wfT     = [(c = Vector{Spline1D}(undef,Nθ),
                a = Vector{Spline1D}(undef,Nθ),
                v = Vector{Spline1D}(undef,Nθ)) for t in 1:T]
    bfT     = [(c = Vector{Spline1D}(undef,Nθ),
                a = Vector{Spline1D}(undef,Nθ),
                v = Vector{Spline1D}(undef,Nθ),
                k = Vector{Spline1D}(undef,Nθ),
                n = Vector{Spline1D}(undef,Nθ),
                y = Vector{Spline1D}(undef,Nθ),
                π = Vector{Spline1D}(undef,Nθ)) for t in 1:T]

    cf_w    = Vector{Spline1D}(undef,Nθ)
    af_w    = Vector{Spline1D}(undef,Nθ)
    Vf_w    = Vector{Spline1D}(undef,Nθ)
    cf_b    = Vector{Spline1D}(undef,Nθ)
    af_b    = Vector{Spline1D}(undef,Nθ)
    Vf_b    = Vector{Spline1D}(undef,Nθ)
    kf      = Vector{Spline1D}(undef,Nθ)
    nf      = Vector{Spline1D}(undef,Nθ)
    yf      = Vector{Spline1D}(undef,Nθ)
    πf      = Vector{Spline1D}(undef,Nθ)

    Vhold   = OCM.Vcoefs 
    luΦ     = lu(Φ)

    
    for t in reverse(1:T)

        OCM.Vcoefs = Vhold
        OCM.r      = rT[t]
        OCM.tr     = trT[t]
        OCM.w      = (1-α)*Θ̄*((rT[t]/(1-τp)+δ)/(Θ̄*α))^(α/(α-1))

        cf_w,af_w = policyw(OCM)
        cf_b,af_b,kf,nf,yf,πf = policyb(OCM)
    
        #Compute values at gridpoints     
        c_w,c_b = zeros(Na,Nθ),zeros(Na,Nθ) 
        a_w,a_b = zeros(Na,Nθ),zeros(Na,Nθ)
        Vw,Vb   = zeros(Na,Nθ),zeros(Na,Nθ)
        for s in 1:Nθ 
            c_w[:,s] = cf_w[s](agrid)
            c_b[:,s] = cf_b[s](agrid)
            a_w[:,s] = af_w[s](agrid)     
            a_b[:,s] = af_b[s](agrid)     
            EΦw      = kron(πθ[s,:]',BasisMatrix(abasis,Direct(),a_w[:,s]).vals[1])
            EΦb      = kron(πθ[s,:]',BasisMatrix(abasis,Direct(),a_b[:,s]).vals[1])
            Vw[:,s]  = c_w[:,s].^(1-σ)/(1-σ) + β.*EΦw*Vhold
            Vb[:,s]  = c_b[:,s].^(1-σ)/(1-σ) + β.*EΦb*Vhold
        end
        p       = probw.(Vb.-Vw,σ_ε)      
        V       = p.*Vw .+ (1 .- p).*Vb
        ptol    = 1e-8
        ip      = ptol.< p .< 1-ptol      
        V[ip]  .= Vw[ip] .+ σ_ε.*log.(1 .+ exp.((Vb[ip].-Vw[ip])./σ_ε))

        #Implied value functions
        Vf_w    = [Spline1D(agrid,Vw[:,s],k=1) for s in 1:Nθ]
        Vf_b    = [Spline1D(agrid,Vb[:,s],k=1) for s in 1:Nθ]
    
        #Update the coefficients using the linear system ΦVcoefs = V
        Vhold  .= luΦ\V[:]                
        wfT[t]  = (c=cf_w,a=af_w,v=Vf_w)
        bfT[t]  = (c=cf_b,a=af_b,v=Vf_b,k=kf,n=nf,y=yf,π=πf)

    end
    return wfT,bfT
end 

function residual_tr!(x0,OCMold,OCMnew)

    @unpack T,Nh,alθ,Ia,Nθ,τc,τp,τd,τw,τb,δ,Θ̄,α,γ,g,b = OCMnew

    #Initialize paths
    rT      = x0[1:T]
    trT     = x0[T+1:2*T]
    ωT      = zeros(Nh,T+1)
    ωT[:,1] = OCMold.ω
    OCMtmp  = deepcopy(OCMnew)
    kres    = zeros(T)
    gres    = zeros(T)

    #Backward: compute policies
    wfT,bfT = policy_path(OCMtmp,rT,trT)

    #Forward: update distribution over time 
    ah      = alθ[1:Ia,1] 
    adst    = hcat([alθ[:,1];alθ[:,1]])
    for t in 1:T

        @views forward_alt!(OCMnew,ωT[:,t],ωT[:,t+1],rT[t],trT[t],wfT[t],bfT[t])

        nb      = hcat([bfT[t].n[s](ah) for s in 1:Nθ]...)
        kb      = hcat([bfT[t].k[s](ah) for s in 1:Nθ]...)
        yb      = hcat([bfT[t].y[s](ah) for s in 1:Nθ]...)
        vw      = hcat([wfT[t].v[s](ah) for s in 1:Nθ]...)
        vb      = hcat([bfT[t].v[s](ah) for s in 1:Nθ]...)
        cw      = hcat([wfT[t].c[s](ah) for s in 1:Nθ]...)
        cb      = hcat([bfT[t].c[s](ah) for s in 1:Nθ]...)
        nwdst   = [exp.(alθ[:,3]);zeros(Ia*Nθ)]
        nbdst   = [zeros(Ia*Nθ);nb[:]]
        kbdst   = [zeros(Ia*Nθ);kb[:]]
        ybdst   = [zeros(Ia*Nθ);yb[:]]
        vdst    = [vw[:];vb[:]]
        cdst    = [cw[:];cb[:]]
        A       = dot(ωT[:,t+1],adst)
        Nb      = dot(ωT[:,t+1],nbdst)
        Nc      = dot(ωT[:,t+1],nwdst)-Nb
        K2Nc    = ((rT[t]/(1-τp)+δ)/(Θ̄*α))^(1/(α-1))
        Kc      = K2Nc*Nc
        w       = (1-α)*Θ̄*K2Nc^α
        Yc      = Θ̄*Kc^α*Nc^(1-α)
        Kb      = dot(ωT[:,t+1],kbdst)
        Yb      = dot(ωT[:,t+1],ybdst)
        C       = dot(ωT[:,t+1],cdst)
        Tc      = τc*C
        Tp      = τp*(Yc-w*Nc-δ*Kc)
        Td      = τd*(Yc-w*Nc-(γ+δ)*Kc-Tp)
        Tw      = τw*w*(Nc+Nb)
        Tb      = τb*(Yb-(rT[t]+δ)*Kb-w*Nb)
        tx      = Tc+Tp+Td+Tw+Tb
        kres[t] = 1-((1-τd)*Kc+Kb+b)/A
        gres[t] = 1-(tx-trT[t]-(rT[t]-γ)*b)/g

    end
    res  = vcat(kres,gres)
    return res
end

function solve_tr!(OCMold::OCModel, OCMnew::OCModel)

    #Initial transition path guess and temporary struct
    x0      = vcat(LinRange(OCMold.r,OCMnew.r,OCMold.T),
                   LinRange(OCMold.tr,OCMnew.tr,OCMold.T))

    #Residuals for asset markets and government budgets
    f!(x)   = residual_tr!(x,OCMold,OCMNew)

    res     = nlsolve(f!,x0; method = :newton, linesearch = :bt)
    x       = res.zero
    rT      = x[1:T]
    trT     = x[T+1:2*T]

end