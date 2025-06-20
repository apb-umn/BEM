# This file has functions for solving the Occupation Choice Model (OCM) using the EGM method.

"""

setup!(OCM::OCModel)

Setup grids and equilibrium guesses
Inputs: parameters (OCM)
Outputs: agrid,alθ,πθ,lθ,Φ,EΦeg,EΦ_aeg,ω,Λ,r,w,tr,Vcoefs

"""
function setup!(OCM::OCModel)

    @unpack a̲,Na,N_θb,N_θw,ρ_θw,σ_θw,bm_θw,ρ_θb,σ_θb,bm_θb,Nθ,
            βo,γ,σ,Ia,amax,so,curv_a,curv_h,trlb,trub,rlb,rub,Θ̄,τp,δ,α,b,g,πθbBM,θbgridBM,πθwBM,θwgridBM,risk_adjust = OCM

    #Productivity shocks of workers
    if bm_θw==0
        mc = rouwenhorst(N_θw,ρ_θw,σ_θw)
        πθw = mc.p
        θwgrid = exp.(mc.state_values)
    else
        πθw = πθwBM
        θwgrid =θwgridBM
    end
    θwgrid =θwgrid.^risk_adjust
    
    #Productivity shocks of entrepreneurs
    if bm_θb==0
        mc = rouwenhorst(N_θb,ρ_θb,σ_θb)
        πθb = mc.p
        θbgrid = exp.(mc.state_values)
    else
        πθb =πθbBM
        θbgrid = θbgridBM
    end
    θbgrid=θbgrid.^risk_adjust

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
    Φₐ = kron(Matrix(I,Nθ,Nθ),BasisMatrix(abasis,Direct(),agrid,[1]).vals[1])

    OCM.EΦeg = kron(πθ,BasisMatrix(abasis,Direct(),agrid).vals[1])
    OCM.EΦ_aeg = kron(πθ,BasisMatrix(abasis,Direct(),agrid,[1]).vals[1])

    #Grid for histogram
    xvec    = LinRange(0,1,Ia).^curv_h
    hagrid  = a̲ .+ (amax - a̲).*xvec 
    OCM.alθ = hcat(kron(ones(Nθ),hagrid),kron(lθ,ones(Ia)))
    OCM.ω   = ones(2*Ia*Nθ)/(2*Ia*Nθ) 

    #Guess prices and transfers
    OCM.r   =r= 0.5*(rlb+rub)
    OCM.tr  =tr= 0.5*(trlb+trub)
    rc     = r/(1-τp)
    K2Nc   = ((rc+δ)/(Θ̄*α))^(1/(α-1))
    OCM.w  = w = (1-α)*Θ̄*K2Nc^α
    OCM.tx  = tr+ g+ (r-γ)*b   

    #Ensure discount factor is updated properly
    OCM.βEE    = βo*(1+γ)^(-σ)
    OCM.βV    = βo*(1+γ)^(1-σ)

    #Guess for unknown coefficients
    c_guess = (1-βo)*agrid .+ w
    OCM.Vcoefs  = Φ\repeat(c_guess.^(1-σ)./(1-σ)/(1-βo),Nθ)
    #λ = Φₐ * OCM.Vcoefs
    OCM.λcoefs  = Φ\ (Φₐ* OCM.Vcoefs)#repeat((1+OCM.r).*c_guess.^(-σ),Nθ)


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

cf,af,λf = policyw(OCM)

EG method applied to worker dynamic program
Inputs: parameters (OCM)
Outputs: policy rules (cf,af,λf)

"""
function policyw(OCM::OCModel)

    @unpack Vcoefs,λcoefs,σ,βEE,γ,Nθ,lθ,a̲,EΦ_aeg,EΦeg,agrid,r,w,tr,τc,τw = OCM

    lθw  = lθ[:,2]
    θw   = exp.(lθw)
    w̄    = (1-τw)*w

    #Compute value function derivative
    EVₐ′ = reshape(EΦeg*λcoefs,:,Nθ)
    EVₐ′ = max.(EVₐ′,1e-6)

    #Compute consumption today implied by Euler equation
    cEE = (βEE.*EVₐ′).^(-1/σ) #consumption today

    #Compute asset today implied by savings and consumtion
    Implieda = ((1+γ).*agrid .+ (1+τc).*cEE .- w̄.*θw' .-tr)./(1+r)  

    #Initialize policy rules for each productivity
    cf = Vector{Spline1D}(undef,Nθ) 
    af = Vector{Spline1D}(undef,Nθ)
    λf = Vector{Spline1D}(undef,Nθ)

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
                λ̂ = (1+r).*ĉ.^(-σ)
                cf[s] = Spline1D(â,ĉ,k=1,bc="extrapolate")
                af[s] = Spline1D(â,[a̲;agrid],k=1,bc="extrapolate")
                λf[s] = Spline1D(â,λ̂,k=1,bc="extrapolate")
            else
                cf[s] = Spline1D(Implieda[:,s],cEE[:,s],k=1,bc="extrapolate")
                af[s] = Spline1D(Implieda[:,s],agrid,k=1,bc="extrapolate")
                λf[s] = Spline1D(Implieda[:,s],(1+r).*cEE[:,s].^(-σ),k=1,bc="extrapolate")
            end
        else
            p = sortperm(Implieda[:,s])
            if Implieda[p[1],s] > a̲ #borrowing constraint binds
                #Add extra points on the borrowing constraint for interpolation
                â = [a̲;Implieda[p,s]]
                ĉ = [((1+r)*a̲-(1+γ)a̲+w̄*θw[s]+tr)/(1+τc);cEE[p,s]]
                cf[s] = Spline1D(â,ĉ,k=1,bc="extrapolate")
                af[s] = Spline1D(â,[a̲;agrid[p]],k=1,bc="extrapolate")
                λf[s] = Spline1D(â,(1+r).*ĉ.^(-σ),k=1,bc="extrapolate")
            else
                cf[s] = Spline1D(Implieda[p,s],cEE[p,s],k=1,bc="extrapolate")    
                af[s] = Spline1D(Implieda[p,s],agrid[p],k=1,bc="extrapolate")
                λf[s] = Spline1D(Implieda[p,s],(1+r).*cEE[p,s].^(-σ),k=1,bc="extrapolate")
            end
        end
    end
    return cf,af,λf
end


"""

cf,af,kf,nf,yf,πf,λf = policyb(OCM)

EG method applied to business dynamic program
Inputs: parameters (OCM), V coefficients (Vcoefs)
Outputs: policy rules (cf,af,kf,nf,yf,πf,ζf,λf)

"""
function policyb(OCM::OCModel)
    @unpack Vcoefs,λcoefs,σ,βEE,γ,Nθ,lθ,a̲,EΦeg,Na,agrid,α_b,ν,δ,χ,r,w,tr,τc,τb,k_min = OCM

    lθb = lθ[:,1]
    θb  = exp.(lθb)

    #Initialize policy rules for each productivity
    cf  = Vector{Spline1D}(undef,Nθ)
    af  = Vector{Spline1D}(undef,Nθ)
    kf  = Vector{Spline1D}(undef,Nθ)
    nf  = Vector{Spline1D}(undef,Nθ)
    yf  = Vector{Spline1D}(undef,Nθ)
    πf  = Vector{Spline1D}(undef,Nθ)
    λf  = Vector{Spline1D}(undef,Nθ)

    #Compute firms profit (ignoring constraints)
    nbyk = ν*(r+δ)/(α_b*w) 
    kvec = @. (w/(ν*θb*nbyk^(ν-1)))^(1/(α_b+ν-1))
    πu   = @. θb*kvec^α_b*(nbyk*kvec)^ν-(r+δ)*kvec-w*(nbyk*kvec)

    #Compute value function derivative
    EVₐ′ = reshape(EΦeg*λcoefs,:,Nθ)#
    EVₐ′ = max.(EVₐ′,1e-6)

    #Compute consumption today implied by Euler equation
    cEE = (βEE.*EVₐ′).^(-1/σ) 

    #Compute asset today implied by savings and consumtion
    Implieda = ((1+γ).*agrid .+ (1+τc).*cEE .- (1-τb).*πu' .-tr) ./ (1+r) 

    #Find out where borrowing constraints bind
    k   = ones(Na).*kvec'
    y   = ones(Na).*(θb.*kvec.^α_b.*(nbyk.*kvec).^ν)'
    πb  = ones(Na).*πu'
    n   = nbyk.*k
    λ   = (1+r).*cEE.^(-σ)

    #Compute the argument for EGM Inverse 
    argEGMinv = (1+γ).*agrid .+  (1+τc).*cEE .- tr
    nmin = (w./(ν.*θb.*k_min.^α_b)).^(1/(ν-1))
    ymin = θb.*k_min.^α_b.*nmin.^ν
    πbmin = ymin .- w.*nmin .- (r+δ).*k_min'

    


    for s in 1:Nθ
        ic   = χ.*Implieda[:,s] .+ k_min .< kvec[s] 
        
        if sum(ic) > 0
            Implieda[ic,s] = OCM.egi[s](argEGMinv[ic,s])
            k[ic,s]  = max.(χ.*Implieda[ic,s].+k_min,k_min)
            n[ic,s]  = (w./(ν.*θb[s].*k[ic,s].^α_b)).^(1/(ν-1))
            y[ic,s]  = θb[s].*k[ic,s].^α_b.*n[ic,s].^ν
            πb[ic,s] = y[ic,s] - w.*n[ic,s] - (r+δ).*k[ic,s]

            mpkc_wedge = α_b*θb[s].*k[ic,s].^(α_b-1).*n[ic,s].^ν .- (r+δ)
            λ[ic,s] = (1+r).*cEE[ic,s].^(-σ) .+ χ.*cEE[ic,s].^(-σ).* mpkc_wedge.*(1-τb)
        end
        kminmask = k[:,s] .<= k_min
        Implieda[kminmask,s] =  (argEGMinv[kminmask,s] .- (1-τb).*πbmin[s]) ./ (1+r)
    end

    #Update where borrowing constraint binding and interpolate
    numa = 10
    for s in 1:Nθ
        min_a=minimum(Implieda[:,s])
        if min_a > a̲ 
            acon  = LinRange(a̲,min_a,numa+1)[1:end-1]#unique([agrid[agrid .<= min_a];min_a])#
            kcon  = min.(χ.*acon .+ k_min,kvec[s])
            ncon  = (w./(ν.*θb[s].*kcon.^α_b)).^(1/(ν-1))
            ycon  = θb[s].*kcon.^α_b.*ncon.^ν
            πcon  = ycon - w.*ncon - (r+δ).*kcon
            ancon = a̲.*ones(length(acon))
            ccon  = ((1+r).*acon .+ (1-τb).*πcon .+ tr .- (1+γ)*a̲) ./(1+τc)
            mpkc_wedge = α_b*θb[s].*kcon.^(α_b-1).*ncon.^ν .- (r+δ)
            λcon  = (1+r).*ccon.^(-σ) .+ χ.*ccon.^(-σ).* mpkc_wedge.*(1-τb)
        end

        if issorted(Implieda[:,s])
            if Implieda[1,s] > a̲ #borrowing constraint binds
                #now interpolate
                cf[s] = Spline1D([acon;Implieda[:,s]],[ccon;cEE[:,s]],k=1,bc="extrapolate")
                af[s] = Spline1D([acon;Implieda[:,s]],[ancon;agrid],k=1,bc="extrapolate")
                kf[s] = Spline1D([acon;Implieda[:,s]],[kcon;k[:,s]],k=1,bc="extrapolate")
                nf[s] = Spline1D([acon;Implieda[:,s]],[ncon;n[:,s]],k=1,bc="extrapolate")
                yf[s] = Spline1D([acon;Implieda[:,s]],[ycon;y[:,s]],k=1,bc="extrapolate")
                πf[s] = Spline1D([acon;Implieda[:,s]],[πcon;πb[:,s]],k=1,bc="extrapolate")
                λf[s] = Spline1D([acon;Implieda[:,s]],[λcon;λ[:,s]],k=1,bc="extrapolate")
            else
                cf[s] = Spline1D(Implieda[:,s],cEE[:,s],k=1,bc="extrapolate")
                af[s] = Spline1D(Implieda[:,s],agrid,k=1,bc="extrapolate")
                kf[s] = Spline1D(Implieda[:,s],k[:,s],k=1,bc="extrapolate")
                nf[s] = Spline1D(Implieda[:,s],n[:,s],k=1,bc="extrapolate")
                yf[s] = Spline1D(Implieda[:,s],y[:,s],k=1,bc="extrapolate")
                πf[s] = Spline1D(Implieda[:,s],πb[:,s],k=1,bc="extrapolate")
                λf[s] = Spline1D(Implieda[:,s],λ[:,s],k=1,bc="extrapolate")
            end
        else
            p = sortperm(Implieda[:,s])
            if Implieda[p[1],s] > a̲ #borrowing constraint binds
                #now interpolate
                cf[s] = Spline1D([acon;Implieda[p,s]],[ccon;cEE[p,s]],k=1,bc="extrapolate")
                af[s] = Spline1D([acon;Implieda[p,s]],[ancon;agrid[p]],k=1,bc="extrapolate")
                kf[s] = Spline1D([acon;Implieda[p,s]],[kcon;k[p,s]],k=1,bc="extrapolate")
                nf[s] = Spline1D([acon;Implieda[p,s]],[ncon;n[p,s]],k=1,bc="extrapolate")
                yf[s] = Spline1D([acon;Implieda[p,s]],[ycon;y[p,s]],k=1,bc="extrapolate")
                πf[s] = Spline1D([acon;Implieda[p,s]],[πcon;πb[p,s]],k=1,bc="extrapolate")
                λf[s] = Spline1D([acon;Implieda[p,s]],[λcon;λ[p,s]],k=1,bc="extrapolate")
            else
                cf[s] = Spline1D(Implieda[p,s],cEE[p,s],k=1,bc="extrapolate")
                af[s] = Spline1D(Implieda[p,s],agrid[p],k=1,bc="extrapolate")
                kf[s] = Spline1D(Implieda[p,s],k[p,s],k=1,bc="extrapolate")
                nf[s] = Spline1D(Implieda[p,s],n[p,s],k=1,bc="extrapolate")
                yf[s] = Spline1D(Implieda[p,s],y[p,s],k=1,bc="extrapolate")
                πf[s] = Spline1D(Implieda[p,s],πb[p,s],k=1,bc="extrapolate")
                λf[s] = Spline1D(Implieda[p,s],λ[p,s],k=1,bc="extrapolate")
            end
        end

    end
    return cf,af,kf,nf,yf,πf,λf
end




"""
egi[s] = setup_egi!(OCM)

Computes the spline inverse in the EG algorithm
Inputs: parameters (OCM)
Outputs: eg inverse for each shock s
"""
function setup_egi!(OCM::OCModel)

    @unpack Nθ,lθ,α_b,ν,χ,r,w,δ,τb,k_min = OCM

    #Compute unconstrained capital
    lθb  = lθ[:,1]
    θb   = exp.(lθb)
    nbyk = ν*(r+δ)/(α_b*w) 
    kvec = @. (w/(ν*θb*nbyk^(ν-1)))^(1/(α_b+ν-1))

    #Compute EG inverse spline
    numk = 100
    curv = 3.
    for s in 1:Nθ
        a_max = (kvec[s]-k_min)/χ
        ahold    = LinRange(0,1,numk).^curv .* a_max
        #ahold    = LinRange(0.,kvec[s]/χ,numk) 
        khold    = ahold.*χ .+ k_min
        nhold    = (w./(ν.*θb[s].*khold.^α_b)).^(1/(ν-1))
        yhold    = θb[s].*khold.^α_b.*nhold.^ν
        πbhold   = yhold - w.*nhold - (r+δ).*khold
        aret     = (1+r).*ahold .+ (1 .- τb).*πbhold
        OCM.egi[s] = Spline1D(aret,ahold,k=2)
    end
end 



"""
V,wf,bf = solve_eg!(OCM)

Solves the functional equation for value function V
Inputs: parameters (OCM)
Outputs: V coefficients (Vcoefs), wf,bf (policies)

"""
function solve_eg!(OCM::OCModel)
 
    @unpack Na,agrid,abasis,σ,βV,Φ,Nθ,lθ,πθ,σ_ε,Nit,tolegm,Nhoward,iprint = OCM


    # Placeholders for splines
    cf_w  = af_w = Vf_w = λf_w = nothing
    cf_b  = af_b = kf = nf = yf = πf = Vf_b = λf_b = nothing

    #Iterate on the value function coefficients
    diff  = 1.
    dchg  = 1.
    tol   = tolegm
    luΦ   = lu(Φ)
    it    = 0
    while diff > tol && it < Nit && dchg > 1e-5

      Vhold = OCM.Vcoefs
      λhold = OCM.λcoefs

      #Compute optimal consumption, asset, and value functions
      cf_w,af_w,λf_w = policyw(OCM)
      cf_b,af_b,kf,nf,yf,πf,λf_b = policyb(OCM)

      for _ =1:Nhoward
        #Compute values at gridpoints
        c_w,c_b = zeros(Na,Nθ),zeros(Na,Nθ) 
        a_w,a_b = zeros(Na,Nθ),zeros(Na,Nθ)
        Vw,Vb   = zeros(Na,Nθ),zeros(Na,Nθ)
        λw,λb   = zeros(Na,Nθ),zeros(Na,Nθ)
        for s in 1:Nθ
            c_w[:,s] = cf_w[s](agrid) 
            c_b[:,s] = cf_b[s](agrid) 
            a_w[:,s] = af_w[s](agrid) 
            a_b[:,s] = af_b[s](agrid)   
            λw[:,s] = λf_w[s](agrid)
            λb[:,s] = λf_b[s](agrid)
            EΦw      = kron(πθ[s,:]',BasisMatrix(abasis,Direct(),a_w[:,s]).vals[1])
            EΦb      = kron(πθ[s,:]',BasisMatrix(abasis,Direct(),a_b[:,s]).vals[1])
            Vw[:,s]  = c_w[:,s].^(1-σ)/(1-σ) + βV.*EΦw*OCM.Vcoefs
            Vb[:,s]  = c_b[:,s].^(1-σ)/(1-σ) + βV.*EΦb*OCM.Vcoefs 
        end
        p       = probw.(Vb.-Vw,σ_ε)
        V       = p.*Vw .+ (1 .- p).*Vb
        λ       = p.*λw .+ (1 .- p).*λb
        ptol    = 1e-8
        ip      = ptol.< p .< 1-ptol
        V[ip]  .= Vw[ip] .+ σ_ε.*log.(1 .+ exp.((Vb[ip].-Vw[ip])./σ_ε))

        #Implied value functions
        Vf_w    = [Spline1D(agrid,Vw[:,s],k=1) for s in 1:Nθ]
        Vf_b    = [Spline1D(agrid,Vb[:,s],k=1) for s in 1:Nθ]

        #Update the coefficients using the linear system ΦVcoefs = V
        Vcnew   = luΦ\V[:]
        λcnew   = luΦ\λ[:]
        OCM.Vcoefs = OCM.λ .* Vcnew + (1-OCM.λ) .* Vhold
        OCM.λcoefs = OCM.λ .* λcnew + (1-OCM.λ) .* λhold
    end 
    diff    = norm(OCM.λcoefs.- λhold,Inf)
    OCM.diffv= diff

      it     += 1
    end

    OCM.wf  = (c=cf_w,a=af_w,v=Vf_w,λ=λf_w)
    OCM.bf  = (c=cf_b,a=af_b,v=Vf_b,k=kf,n=nf,y=yf,π=πf,λ=λf_b)
    if iprint==1
        
        if it>= Nit
            println("solve_eg did not converge: $diff")
        else
            println("solve_eg converged in $it iterations: $diff")
        end
        if dchg<=1e-5
            println("solve_eg is making no progress: $dchg")
        end
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
        #setup!(OCM)
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
        Y= Yc+Yb
        Tp     = τp*(Yc-w*Nc-δ*Kc)
        Td     = τd*(Yc-w*Nc-(γ+δ)*Kc-Tp)
        Tw     = τw*w*(Nc+Nb)
        Tb     = τb*(Yb-(r+δ)*Kb-w*Nb)
        OCM.tx = Tc+Tp+Td+Tw+Tb
        res    = [A-((1-τd)*Kc+Kb+b),g-(OCM.tx-tr-(OCM.r-γ)*b)]/Y

        if iprint==1
            @printf("      Tax on business %10.3f, Tax on wages %10.3f\n",τb*100,τw*100)
            @printf("      Interest rate %10.3f, Govt transfer %10.3f\n",r*100,tr)
            @printf("      Asset market  %10.3e, Govt budget   %10.3e\n",res[1],res[2])
            println("")
        end
        return res[1:2]

    end

    if ibise==1
        @printf("      Using bisection method to get a good guess\n")
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
        println("      Solve SS using newton method")
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



"""
    weighted_quantile(x, w, p)

Compute the weighted quantile(s) of `x` with weights `w` at probability `p`.
- `x`: data vector (e.g. lnwdst)
- `w`: weights (e.g. OCM.ω)
- `p`: quantile level(s), e.g. 0.10 or [0.10, 0.90]
"""
function weighted_quantile(x::AbstractVector, w::AbstractVector, p::Union{Real,AbstractVector})
    idx = sortperm(x)
    x_sorted = x[idx]
    w_sorted = w[idx]
    cum_weights = cumsum(w_sorted)
    total_weight = sum(w_sorted)
    probs = cum_weights ./ total_weight

    # Helper to find value at quantile q
    function qval(q)
        i = findfirst(>=(q), probs)
        return x_sorted[i]
    end

    return isa(p, Number) ? qval(p) : map(qval, p)
end


function getMoments(OCM::OCModel;savepath::String="macro_ratios.tex")
     @unpack τp,τd,τb,τc,τw,δ,Θ̄,α,b,γ,g,iprint,r,tr,bf,wf,Ia,Nθ,alθ,ab_col_cutoff,lθ,χ = OCM
    updatecutoffs!(OCM)
    rc     = r/(1-τp)
    K2Nc   = ((rc+δ)/(Θ̄*α))^(1/(α-1))
    w = (1-α)*Θ̄*K2Nc^α
    ah  = alθ[1:Ia,1] #grids are all the same for all shocks

    nb     = hcat([bf.n[s](ah) for s in 1:Nθ]...) #labor demand from business
    kb     = hcat([bf.k[s](ah) for s in 1:Nθ]...)  #capital demand from business
    yb     = hcat([bf.y[s](ah) for s in 1:Nθ]...) #value added from business
    vw     = hcat([wf.v[s](ah) for s in 1:Nθ]...) #welfare of workers
    vb     = hcat([bf.v[s](ah) for s in 1:Nθ]...) #welfare of business owners
    cw     = hcat([wf.c[s](ah) for s in 1:Nθ]...) #consumption of workers
    cb     = hcat([bf.c[s](ah) for s in 1:Nθ]...) #consumption of business owners
    pib    =hcat([bf.π[s](ah) for s in 1:Nθ]...) #profits of business owners
    lnwdst  = [log(OCM.w).+alθ[:,3];zeros(Ia*Nθ)] #mlog wage earnings
    nwdst  = [exp.(alθ[:,3]);zeros(Ia*Nθ)] #
    zdst = hcat([zeros(Ia*Nθ);exp.(alθ[:,2])]) #productivity shocks


    nbdst  = [zeros(Ia*Nθ);nb[:]]
    pidst  = [zeros(Ia*Nθ);pib[:]] #profits of business owners
    kbdst  = [zeros(Ia*Nθ);kb[:]]
    mpkdist = OCM.α_b.*zdst.*kbdst.^(OCM.α_b-1).*nbdst.^OCM.ν
   lnmpkdist = log.(mpkdist)

    ybdst  = [zeros(Ia*Nθ);yb[:]]
    vdst   = [vw[:];vb[:]]
    cdst   = [cw[:];cb[:]]
    adst   = hcat([alθ[:,1];alθ[:,1]])
    awdist= hcat([alθ[:,1];alθ[:,1].*0])
    abdist = hcat([alθ[:,1].*0;alθ[:,1]]) 
    indcons = hcat([(ah .< ab_col_cutoff[lθ[s,:]]) for s in 1:Nθ]...)
    indconsdist=[zeros(Ia*Nθ);indcons[:]]
    acons = hcat([ah.*(ah .< ab_col_cutoff[lθ[s,:]]) for s in 1:Nθ]...)
    aconsdist=[zeros(Ia*Nθ);acons[:]]
    kbcons = hcat([kb[:,s].*(ah .< ab_col_cutoff[lθ[s,:]]) for s in 1:Nθ]...)
    kbconsdist=[zeros(Ia*Nθ);kbcons[:]]


    
    Nb     = dot(OCM.ω,nbdst) #agg labor demand from business
    Nc     = dot(OCM.ω,nwdst)-Nb #agg labor demand from corporate sector
    wN      =w*(Nc+Nb) #agg wage bill
    wNc    = w*Nc #wage bill from corporate sector
    wNb    = w*Nb #wage bill from business
    Kc     = K2Nc*Nc  #capital demand from corporate sector
    Yc     = Θ̄*Kc^α*Nc^(1-α) #value added from corporate sector
    Kb     = dot(OCM.ω,kbdst) #agg capital demand from business
    Yb     = dot(OCM.ω,ybdst) #agg value added from business
    Pib     =Yb-(r+δ)*Kb-w*Nb
    rcKc      =Yc-w*Nc-δ*Kc
    rKb    = r*Kb
    dK    = δ*Kc+δ*Kb #depreciation of capital
    dgK   =(γ+δ)*(Kc+Kb) #invest



    C      = dot(OCM.ω,cdst) #agg consumption
    Tc     = τc*C #tax on consumption
    Tp     = τp*(Yc-w*Nc-δ*Kc) #tax on profits of corporate sector
    Td     = τd*(Yc-w*Nc-(γ+δ)*Kc-Tp) #tax on dividends of business
    Tn     = τw*w*(Nc+Nb) #tax on labor income
    Tb     = τb*(Yb-(r+δ)*Kb-w*Nb) #tax on profits of business
    tx = Tc+Tp+Td+Tn+Tb #total tax revenue
    G=g #government spending
    T= tr #government transfers
    iB =(r-γ)*b
    GTiB = G+T+iB #government budget



    Frac_b =sum(reshape(OCM.ω,:,2),dims=1)[2] # fraction of borrowing agents
    V = dot(OCM.ω,vdst) #average utility
    A = dot(OCM.ω,adst) # average assets
    Aw= dot(OCM.ω,awdist) # average assets of workers
    Ab= dot(OCM.ω,abdist) # average assets of business owners
    C      = dot(OCM.ω,cdst) # average consumption
    K=Kb+Kc # total capital in the economy
    Y=Yb+Yc # total output in the economy


    # collateral constraints
    Frac_b_cons= dot(OCM.ω,indconsdist) # fraction of constrained agents

    Acons= dot(OCM.ω,aconsdist) # average assets of constrained agents

    Kbcons= dot(OCM.ω,kbconsdist) # average capital of constrained agents

    Bb=Kbcons-Acons # external debt of business owners

    Bb_by_Kb = Bb/Kb # external debt to capital ratio of business owners
    Bb_by_Y = Bb/Y # external debt of private to agg. gdp of business owners


    #distribuitional moments
    indx_workers=1:Ia*Nθ # indices for business owners
    workers_ω = OCM.ω[indx_workers]./sum(OCM.ω[indx_workers]) # weights for workers
    mean_log_wageearnings = dot(workers_ω,lnwdst[indx_workers]) # mean log wage earnings
    std_log_wageearnings = sqrt(dot(workers_ω,((lnwdst[indx_workers] .- mean_log_wageearnings).^2))) # std log wage earnings
    q10_log_wageearnings, q25_log_wageearnings, q75_log_wageearnings, q90_log_wageearnings = weighted_quantile(lnwdst[indx_workers], workers_ω, [0.10, 0.25,0.75,0.90])
    
    indx_owners=Ia*Nθ+1:Ia*2*Nθ # indices for workers
    owners_ω = OCM.ω[indx_owners]./sum(OCM.ω[indx_owners]) # weights for business owners
    sel=.!isnan.(lnmpkdist[indx_owners])
    mean_mpkdist = dot(owners_ω[sel],lnmpkdist[indx_owners][sel])
    std_mpkdist = sqrt(dot(owners_ω[sel],((lnmpkdist[indx_owners][sel] .- mean_mpkdist).^2))) # std log MPK 
    q10_mpk, q25_mpk, q75_mpk, q90_mpk = weighted_quantile(lnmpkdist[indx_owners][sel], owners_ω, [0.10, 0.25,0.75,0.90])
    piybdst =pidst[indx_owners]./(ybdst[indx_owners] .+ .0001)
    mean_piybdst = dot(owners_ω,piybdst) # mean profit share
    std_piybdst = sqrt(dot(owners_ω,((piybdst .- mean_piybdst).^2))) # std profit share
    pidst_plus1 = max.(pidst[indx_owners], 0.0001) # avoid division by zero
    lnpidst= log.(pidst_plus1) # log profit share
    q10_lnpi, q25_lnpi, q75_lnpi, q90_lnpi = weighted_quantile(lnpidst, owners_ω, [0.10, 0.25,0.75,0.90])
    mean_lnpi=dot(owners_ω,lnpidst) # mean profit share
    std_lnpi = sqrt(dot(owners_ω,((lnpidst .- mean_lnpi).^2))) # std log profit share

# ratios
    Y_Y = Y/Y # output to output ratio (should be 1)
    WN_Y = wN/Y # wage bill to output ratio
    WNc_Y = wNc/Y # wage bill to output ratio of corporate sector
    WNb_Y = wNb/Y # wage bill to output ratio of business
    Pib_Y = Pib/Y # profit share of business owners
    rKb_Y = rKb/Y # interest payments to output ratio of business
    rKc_Y = rcKc/Y # interest payments to output ratio of corporate sector
    rK_Y = (rKb+rcKc)/Y # interest payments to output ratio of total capital
    dK_Y = dK/Y # depreciation to output ratio

    C_Y = C/Y # consumption to output ratio
    G_Y = G/Y # government spending to output ratio
    dgK_Y = dgK/Y # investment to output ratio
    
    Tax_Y = tx/Y # tax revenue to output ratio
    Tn_Y = Tn/Y # tax on labor income to output ratio
    Tb_Y = Tb/Y # tax on profits of business to output ratio
    Tp_Y = Tp/Y # tax on profits of corporate sector to output ratio
    Tc_Y = Tc/Y # tax on consumption to output ratio
    GTiB_Y = GTiB/Y # government budget to output ratio
    T_Y = T/Y # government transfers to output ratio
    iB_Y = iB/Y # interest payments to output ratio of government debt
    A_Y = A/Y # assets to output ratio
    Ab_Y = Ab/Y # assets of business owners to output ratio
    Aw_Y = Aw/Y # assets of workers to output ratio
    Bb_by_Y = Bb/Y # external debt to output ratio of business owners
    Nfc =Frac_b_cons/Frac_b
    Kfc=Kbcons/Kb


# Create 2-column matrix explicitly
moments = [
    "tax on biz profits (Tb)"     τb;
    "tax on labor income (Tn)"    τw;
    "collateral constraint (χ)"  χ;
    "interest rate (r)"           r;
    "govt transfer (tr)"          tr;
    "wage (w)"                w;
    "Nb (pvt biz labor demand)"       Nb;
    "Nc (corp labor demand)"      Nc;
    "Kc (corp capital demand)"    Kc;
    "Yc (corp value added)"       Yc;
    "Kb (pvt biz capital demand)"     Kb;
    "Yb (pvt biz value added)"        Yb;
    "C (consumption)"             C;
    "K (total capital)"          K;
    "Y (total output)"           Y;
    "Tc (tax on cons.)"           Tc;
    "Tp (tax on profits corp)"    Tp;
    "Tn (labor income tax)"       Tn;
    "Tb (tax on profits pvt biz)"     Tb;
    "Total tax revenue"           tx;
    "Fraction of biz owners"       Frac_b;
    "total. utility (V)"            V;
    "total. assets (A)"             A;
    "total. assets (workers)"       Aw;
    "total. assets (owners)"        Ab;
    "total. consumption (C)"        C;
    "Fraction constrained"        Frac_b_cons;
    "Assets constrained"          Acons;
    "Capital constrained"         Kbcons;
    "External debt (Bb)"          Bb;
    "Bb/Kb (debt to capital ratio)" Bb_by_Kb;
    "Bb/Y (debt to output ratio)" Bb_by_Y;
    "mean log wage earnings"      mean_log_wageearnings;
    "std log wage earnings"       std_log_wageearnings;
    "10% quantile log wage earnings" q10_log_wageearnings;
    "25% quantile log wage earnings" q25_log_wageearnings;
    "75% quantile log wage earnings" q75_log_wageearnings;
    "90% quantile log wage earnings" q90_log_wageearnings;
    "mean log MPK"                mean_mpkdist;
    "std log MPK"                 std_mpkdist;
    "mean profit share"           mean_piybdst;
    "std profit share"            std_piybdst;
    "10% quantile log profits"    q10_lnpi;
    "25% quantile log profits"    q25_lnpi;
    "75% quantile log profits"    q75_lnpi;
    "90% quantile log profits"    q90_lnpi;
    "mean log profits"            mean_lnpi;
    "std log profits"             std_lnpi;

];

# Print it as a table
pretty_table(moments; header=["Moment", "Value"], formatters=ft_printf("%.4f"),crop = :none)


export_macro_ratios_unscaled(savepath; (;Y_Y, WN_Y, WNc_Y, WNb_Y, Pib_Y,
    rK_Y, rKc_Y, rKb_Y, dK_Y,
    C_Y, G_Y, dgK_Y,
    Tax_Y, Tn_Y, Tb_Y, Tp_Y, Tc_Y,
    GTiB_Y, T_Y, iB_Y,
    A_Y, Ab_Y, Aw_Y, Bb_by_Y,
    Nfc, Kfc)...)

    return moments
end

function compare_moments(OCM_old::OCModel, OCM_new::OCModel)

    moments1=getMoments(OCM_old)
    moments2=getMoments(OCM_new)
    # Ensure both matrices have the same number of rows
    if size(moments1, 1) != size(moments2, 1)
        error("The number of moments in the two models do not match.")
    end
    
    # Extract labels and values from each matrix
    labels = moments1[:, 1]               # keep the first column of either (they're same)
    vals1 = Float64.(moments1[:, 2])      # ensure numerical type
    vals2 = Float64.(moments2[:, 2])      # same for second run
    
    # Optional: % difference
    pct_diff = 100 .* (vals2 .- vals1) ./ abs.(vals1)
    
    # Combine into one matrix: Moment | Run 1 | Run 2 | % Diff
    combined = hcat(labels, vals1, vals2, pct_diff)
    
    # Display
    pretty_table(combined; header=["Moment", "Run 1", "Run 2", "% Diff"], formatters=ft_printf("%.4f"))
    
       end
    

function assign!(OCM::OCModel,r::Float64,tr::Float64)

    setup!(OCM)
    OCM.r  = r
    OCM.tr = tr

    @unpack τp,τd,τb,τc,τw,δ,Θ̄,α,b,γ,g,iprint = OCM

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
    diffv=OCM.diffv
    if iprint ==1
        @printf("  Asset market  %10.3e, Govt budget   %10.3e\n",res[1],res[2])
        @printf("  Diff in EGM     %10.3e\n",diffv)
    end

end


function updatecutoffs!(OCM::OCModel)

    @unpack alθ,Ia,Nθ,lθ, χ,agrid,  a̲, Na,Ia, ab_col_cutoff,ab_bor_cutoff, aw_bor_cutoff, bf,wf, k_min = OCM
    
    # where do constraints bind?
    ah  = agrid #grids are all the same for all shocks
    ah=alθ[1:Ia,1]
    kb     = hcat([bf.k[s](ah) for s in 1:Nθ]...)

    # for each shock, find the borrowing constraint
    for s in 1:Nθ
        indices = findall(kb[:,s] .≈ ah*χ .+ k_min)
        if !isempty(indices)
            ab_col_cutoff[lθ[s, :]] = maximum(indices) == 1 ? -Inf : ah[maximum(indices)]
        else
            ab_col_cutoff[lθ[s, :]] = -Inf  # or another default value
        end
    end


    # borrowing constraint for owners
    a′ = hcat([bf.a[s](ah) for s in 1:Nθ]...)
    for s in 1:Nθ
        indices = findall(a′[:, s] .> 0)
        if !isempty(indices)
            ab_bor_cutoff[lθ[s, :]] = minimum(indices) == 1 ? -Inf : ah[minimum(indices) - 1]
        else
            ab_bor_cutoff[lθ[s, :]] = -Inf  # or another fallback value
        end
    end

    # borrowing constraint for workers
    a′ = hcat([wf.a[s](ah) for s in 1:Nθ]...)
    for s in 1:Nθ
        indices = findall(a′[:, s] .> 0)
        if !isempty(indices)
            aw_bor_cutoff[lθ[s, :]] = minimum(indices) == 1 ? -Inf : ah[minimum(indices) - 1]
        else
            aw_bor_cutoff[lθ[s, :]] = -Inf  # or another fallback value
        end
    end


end


function forward!(OCM,ω,ωn,r,tr,τb, wf,bf)

    @unpack Nθ,πθ,lθ,Ia,alθ,δ,Θ̄,α,τp,τc,τw,γ,σ_ε = OCM

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


    
function policy_path(OCM,rT,trT,τbT)

    @unpack Na,agrid,abasis,σ,βEE,βV,Φ,Nθ,lθ,πθ,σ_ε,α,Θ̄,τp,δ = OCM

    T       = length(rT)
    wfT     = [(c = Vector{Spline1D}(undef,Nθ),
                a = Vector{Spline1D}(undef,Nθ),
                v = Vector{Spline1D}(undef,Nθ),
                λ = Vector{Spline1D}(undef,Nθ)) for t in 1:T]
    bfT     = [(c = Vector{Spline1D}(undef,Nθ),
                a = Vector{Spline1D}(undef,Nθ),
                v = Vector{Spline1D}(undef,Nθ),
                k = Vector{Spline1D}(undef,Nθ),
                n = Vector{Spline1D}(undef,Nθ),
                y = Vector{Spline1D}(undef,Nθ),
                π = Vector{Spline1D}(undef,Nθ),
                λ = Vector{Spline1D}(undef,Nθ)) for t in 1:T]

    c_w,c_b = zeros(Na,Nθ),zeros(Na,Nθ) 
    a_w,a_b = zeros(Na,Nθ),zeros(Na,Nθ)
    Vw,Vb   = zeros(Na,Nθ),zeros(Na,Nθ)
    λw,λb   = zeros(Na,Nθ),zeros(Na,Nθ)
    kf      = Vector{Spline1D}(undef,Nθ)
    nf      = Vector{Spline1D}(undef,Nθ)
    yf      = Vector{Spline1D}(undef,Nθ)
    πf      = Vector{Spline1D}(undef,Nθ)


    Vhold = OCM.Vcoefs
    λhold = OCM.λcoefs
    luΦ     = lu(Φ)

    
    for t in reverse(1:T)

        OCM.Vcoefs = Vhold
        OCM.λcoefs = λhold
        OCM.r      = rT[t]
        OCM.tr     = trT[t]
        OCM.w      = (1-α)*Θ̄*((rT[t]/(1-τp)+δ)/(Θ̄*α))^(α/(α-1))
        OCM.τb    = τbT[t]

        cf_w,af_w,λf_w = policyw(OCM)
        cf_b,af_b,kf,nf,yf,πf,λf_b = policyb(OCM)


        
        #Compute values at gridpoints     
        c_w,c_b = zeros(Na,Nθ),zeros(Na,Nθ) 
        a_w,a_b = zeros(Na,Nθ),zeros(Na,Nθ)
        Vw,Vb   = zeros(Na,Nθ),zeros(Na,Nθ)
        for s in 1:Nθ
            c_w[:,s] = cf_w[s](agrid) 
            c_b[:,s] = cf_b[s](agrid) 
            a_w[:,s] = af_w[s](agrid) 
            a_b[:,s] = af_b[s](agrid)   
            λw[:,s] = λf_w[s](agrid)
            λb[:,s] = λf_b[s](agrid)
            EΦw      = kron(πθ[s,:]',BasisMatrix(abasis,Direct(),a_w[:,s]).vals[1])
            EΦb      = kron(πθ[s,:]',BasisMatrix(abasis,Direct(),a_b[:,s]).vals[1])
            Vw[:,s]  = c_w[:,s].^(1-σ)/(1-σ) + βV.*EΦw*OCM.Vcoefs
            Vb[:,s]  = c_b[:,s].^(1-σ)/(1-σ) + βV.*EΦb*OCM.Vcoefs 
        end
        p       = probw.(Vb.-Vw,σ_ε)
        V       = p.*Vw .+ (1 .- p).*Vb
        λ       = p.*λw .+ (1 .- p).*λb 
        ptol    = 1e-8
        ip      = ptol.< p .< 1-ptol
        V[ip]  .= Vw[ip] .+ σ_ε.*log.(1 .+ exp.((Vb[ip].-Vw[ip])./σ_ε))

        #Implied value functions
        Vf_w    = [Spline1D(agrid,Vw[:,s],k=1) for s in 1:Nθ]
        Vf_b    = [Spline1D(agrid,Vb[:,s],k=1) for s in 1:Nθ]

        #Update the coefficients using the linear system ΦVcoefs = V
        Vhold  .= luΦ\V[:]
        λhold  .= luΦ\λ[:]                
        wfT[t]  = (c=cf_w,a=af_w,v=Vf_w,λ=λf_w)
        bfT[t]  =(c=cf_b,a=af_b,v=Vf_b,k=kf,n=nf,y=yf,π=πf,λ=λf_b)

    end
    return wfT,bfT
end 

function residual_tr!(x0,OCMold,OCMnew,τbT)

    @unpack T,Nh,alθ,Ia,Nθ,τc,τp,τd,τw,δ,Θ̄,α,γ,g,b = OCMnew

    #Initialize paths
    rT      = x0[1:T]
    trT     = x0[T+1:2*T]
    ωT      = zeros(Nh,T+1)
    OCMtmp  = deepcopy(OCMnew)
    kres    = zeros(T)
    gres    = zeros(T)

    #Backward: compute policies
    wfT,bfT = policy_path(OCMtmp,rT,trT,τbT)
    ωT[:,1]=reshape_ω0(OCMtmp, OCMold.ω,wfT[1],bfT[1])

    #Forward: update distribution over time 
    ah      = alθ[1:Ia,1] 
    adst    = hcat([alθ[:,1];alθ[:,1]])
    for t in 1:T

        @views forward!(OCMnew,ωT[:,t],ωT[:,t+1],rT[t],trT[t],τbT[t],wfT[t],bfT[t])

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
        A       = dot(ωT[:,t],adst)
        Nb      = dot(ωT[:,t],nbdst)
        Nc      = dot(ωT[:,t],nwdst)-Nb
        K2Nc    = ((rT[t]/(1-τp)+δ)/(Θ̄*α))^(1/(α-1))
        Kc      = K2Nc*Nc
        w       = (1-α)*Θ̄*K2Nc^α
        Yc      = Θ̄*Kc^α*Nc^(1-α)
        Kb      = dot(ωT[:,t],kbdst)
        Yb      = dot(ωT[:,t],ybdst)
        C       = dot(ωT[:,t],cdst)
        Y       = Yc + Yb
        Tc      = τc*C
        Tp      = τp*(Yc-w*Nc-δ*Kc)
        Td      = τd*(Yc-w*Nc-(γ+δ)*Kc-Tp)
        Tw      = τw*w*(Nc+Nb)
        Tb      = τbT[t]*(Yb-(rT[t]+δ)*Kb-w*Nb)
        tx      = Tc+Tp+Td+Tw+Tb
        kres[t] = (A-((1-τd)*Kc+Kb+b))/Y
        gres[t] = (g-(tx-trT[t]-(rT[t]-γ)*b))/Y

    end
    res  = vcat(kres,gres)
    println("Residuals: Asset markets: %.2e, Government budget: %.2e\n", norm(kres), norm(gres))
    return res
end

function solve_tr!(x0,OCMold::OCModel, OCMnew::OCModel,τbT::Vector{Float64})

    #Initial transition path guess and temporary struct
    #x0      = vcat(LinRange(OCMold.r,OCMnew.r,OCMold.T),
    #               LinRange(OCMold.tr,OCMnew.tr,OCMold.T))

    #Residuals for asset markets and government budgets
    f!(x)   = residual_tr!(x,OCMold,OCMnew,τbT)
    res     = nlsolve(f!,x0; method = :newton, linesearch = :bt)
    x       = res.zero
    rT      = x[1:T]
    trT     = x[T+1:2*T]
end



function reshape_ω0(OCM,ω0,wf,bf)
    @unpack Nθ,πθ,lθ,Ia,alθ,δ,Θ̄,α,τp,τc,τb,τw,γ,σ_ε = OCM     
    ω0_reshaped=  reshape(ω0,:,2)
    ω0_aθ = sum(ω0_reshaped, dims=2)
    ah     = alθ[1:Ia,1] 
    Vw     = hcat([wf.v[s](ah) for s in 1:Nθ]...)
    Vb     = hcat([bf.v[s](ah) for s in 1:Nθ]...)
    p      = probw.(Vb.-Vw,σ_ε)

    B = Basis(SplineParams(ah, 0, 1))
    Pw = spdiagm(p[:])                          # Diagonal matrix for worker choice probabilities
    Pb = spdiagm(1.0 .- p[:])           # Diagonal matrix for business owner choice probabilities
    P = vcat(Pw, Pb)  # Combin

    ω_0_new= P*ω0_aθ
    return ω_0_new
end
 


function export_macro_ratios_unscaled(filepath::String;
    Y_Y::Float64,
    WN_Y::Float64, WNc_Y::Float64, WNb_Y::Float64, Pib_Y::Float64,
    rK_Y::Float64, rKc_Y::Float64, rKb_Y::Float64, dK_Y::Float64,
    C_Y::Float64, G_Y::Float64, dgK_Y::Float64,
    Tax_Y::Float64, Tn_Y::Float64, Tb_Y::Float64, Tp_Y::Float64, Tc_Y::Float64,
    GTiB_Y::Float64, T_Y::Float64, iB_Y::Float64,
    A_Y::Float64, Ab_Y::Float64, Aw_Y::Float64, Bb_by_Y::Float64,
    Nfc::Float64, Kfc::Float64
)

    expressions = [
        "Y", "WN", "WNc", "WNb", "Pib",
        "rK", "rcKc", "rKb", "dK",
        "Y", "C", "G", "dgK",
        "Tax", "Tw", "Tb", "Tp", "Tc",
        "GTiB", "G", "T", "iB",
        "A", "Ab", "Aw", "Lb",
        "Nfc", "Kfc"
    ]

    raw_values = [
        Y_Y, WN_Y, WNc_Y, WNb_Y, Pib_Y,
        rK_Y, rKc_Y, rKb_Y, dK_Y,
        Y_Y, C_Y, G_Y, dgK_Y,
        Tax_Y, Tn_Y, Tb_Y, Tp_Y, Tc_Y,
        GTiB_Y, G_Y, T_Y, iB_Y,
        A_Y/100, Ab_Y/100, Aw_Y/100, Bb_by_Y,
        Nfc , Kfc
    ]

    values = round.([x * 100 for x in raw_values], digits=1)

    descriptions = [
        "GDI", "Comp", "CompCC", "CompPB", "Sweat",
        "NOS", "NOSCC", "NOSPB", "Depr",
        "GDP", "Cons", "Defense", "Inve",
        "GRev", "Taxw", "Taxb", "Taxp", "Taxc",
        "GExp", "Defense", "Trans", "Netint",
        "Wealth", "WealthB", "WealthW", "LoansB",
        "NumCon", "CapCon"
    ]

    df = DataFrame(
        Row = 1:28,
        Description = descriptions,
        Expression = expressions,
        Value = values
    )

    CSV.write(filepath, df)
    return df
end
