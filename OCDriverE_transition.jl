# OCDriverE.jl
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
#   Revised, ERM, 5/23/2025


include("OCModelE.jl")
include("FirstOrderApproximation.jl")




function Fw(OCM::OCModel,lθ,a_,x,X,yᵉ)
    @unpack aw_bor_cutoff,β,σ,a̲,τc,τw,γ = OCM
    a_ = a_[1]
    #unpack variables 
    a,n,k,yb,nb,profit,b,λ,v  = x
    λᵉ,vᵉ = yᵉ
    R,W,Tr  = X
    _,ϵ = exp.(lθ)

    u_c = λ/R 
    c = (u_c)^(-1/σ) #definition of λ
    u = c^(1-σ)/(1-σ)
    ret = [R*a_+(1-τw)*W*ϵ+Tr-(1+τc)*c-(1+γ)*a,
           β*λᵉ - u_c,
           u+β*vᵉ - v,
           n + ϵ, #labor supplied so opposite sign
           k,
           yb,
           nb,
           profit,
           b]
    if a_ <= aw_bor_cutoff[lθ] #check if agent is on borrowing constraint
        ret[2] = a̲-a
    end
    return ret
end



function Fb(OCM::OCModel,lθ,a_,x,X,yᵉ)
    @unpack ab_bor_cutoff,ab_col_cutoff,β,σ,α_b,ν,χ,a̲,δ,τb,τc,τw,γ,α_b = OCM
    a_ = a_[1] #deal with vector if needed
    #unpack variables
    a,n,k,yb,nb,profit,b,λ,v  = x
    λᵉ,vᵉ = yᵉ
    R,W,Tr  = X
    r = R-1
    z,_ = exp.(lθ)
    mpk = α_b*z*k^(α_b-1)*n^ν
    mpn = ν*z*k^α_b*n^(ν-1)
    u_c = λ/R 
    c = (u_c)^(-1/σ) #definition of λ
    u = c^(1-σ)/(1-σ)
    ret = [R*a_+(1-τb)*profit+Tr-(1+τc)*c-(1+γ)*a,
           β*λᵉ - u_c,
           u+β*vᵉ - v,
           mpn-W,
           nb-n,
           mpk-r-δ,
           yb - z*k^α_b*n^ν,
           profit  - (z*k^α_b*n^ν - δ*k - r*k - W*n),
           b-1]

    if a_ <=ab_bor_cutoff[lθ] #assuming that if the borrowing constraint binds, then the collateral constraint also binds
        ret[2] = a̲-a
        ret[6] = k-χ*a_
    elseif (a_>ab_bor_cutoff[lθ]) && (a_<= ab_col_cutoff[lθ]) && (a_>0)#check if agent is on collateral constraint
        ret[6] = k-χ*a_
    end

    if a_==0
        ret[4] = n
        ret[5] = nb
        ret[6] = k
        ret[7]=  yb
        ret[8] = profit

    end

     return ret
end


function G(para::OCModel, Ix, A_, X, Xᵉ, Θ)
    @unpack α, δ, τw, τb, τp, τd, τc, b, w, γ, r, g = para
    Ia, In, Ik, Iyb, Inb, Iprofit, Ib, _, Iv = Ix
    R, W, Tr, Frac_b, V, A, C = X
    TFP = Θ[1]
    A_ = A_[1]
    B = b
    Gval = g

    # Corporate sector
    Nc = .-In
    Nb = Inb
    Kc_ = (A_ .- Ik .- B) ./ (1 - τd)
    MPKc = α .* TFP .* Kc_.^(α .- 1) .* Nc.^(1 .- α)
    MPNc = (1 .- α) .* TFP .* Kc_.^α .* Nc.^(-α)
    Yc = TFP .* Kc_.^α .* Nc.^(1 .- α)

    # business sector
    Yb = Iyb
    Kb = Ik

    # aggregates
    K_ = Kc_ .+ Kb
    Kc = (A .- Ik .- B) ./ (1 - τd)
    K  = Kc .+ Kb 
    I = (1+γ)*K .- (1 .- δ) .* K_
    Rc = (R .- 1.0) ./ (1.0 .- τp) .+ 1.0

    # Tax components
    Tp = τp .* (Yc .- w .* Nc .- δ .* Kc_)
    Td = τd .* (Yc .- w .* Nc .- (γ .+ δ) .* Kc_ .- Tp)
    Tn = τw .* w .* (Nc .+ Nb)
    Tb = τb .* (Yb .- (r .+ δ) .* Kb .- w .* Nb)

    return [
        Rc .- 1 .+ δ .- MPKc,
        W .- MPNc,
        A .- Ia,
        C .+ I .+Gval .- Yc .- Yb,
        (τc .* C .+ Tp .+ Td .+ Tn .+ Tb) .- B .* (R .- 1.0 .- γ) .- Gval .-Tr,
        Ib .- Frac_b,
        Iv .- V
    ]
end



function ff(para::OCModel,x⁻,x⁺) #do the discrete choice part
    @unpack σ_ε = para
    _,_,_,_,_,_,_,λ⁻,v⁻  = x⁻
    _,_,_,_,_,_,_,λ⁺,v⁺  = x⁺

    p = 1/(1+exp((v⁺-v⁻)/σ_ε))
    if p<1e-9
        return [λ⁺,v⁺]
    elseif p>1-1e-9
        return [λ⁻,v⁻]
    else
        Ev = v⁻ + σ_ε*log(1+exp((v⁺-v⁻)/σ_ε))  #σ_ε*log(exp(v⁺/σ_ε)+exp(v⁻/σ_ε))
        return [λ⁻*p+λ⁺*(1-p),Ev]
    end
end

function construct_inputs(OCM)
    #populate ithe inputs object
    inputs = Inputs()
    #idiosyncratic objects
    inputs.xf = get_policy_functions(OCM)
    inputs.aknots,inputs.ka,inputs.aθc_sp,inputs.aθc_Ω,inputs.ℵ = get_grids(OCM)
    inputs.xlab=[:a,:n,:k,:yb,:nb,:profit,:b,:λ,:v]
    inputs.alab=[:a]
    inputs.κlab=[:v]
    inputs.yᵉlab=[:λ,:v]
    inputs.Γf = κ-> 1/(1+exp(κ/OCM.σ_ε)) #CDF for Gumbel Distribution
    inputs.dΓf = κ -> -(1/OCM.σ_ε)*exp(κ/OCM.σ_ε)/(1+exp(κ/OCM.σ_ε))^2

    #Aggregate objects
    inputs.X̄=    getX(OCM)
    inputs.Xlab = [:R,:W,:Tr,:Frac_b,:V,:A,:C]
    inputs.Alab = [:A]
    inputs.Qlab = [:R,:W,:Tr]

    #Distributional objects
    inputs.ω̄, inputs.Λ, inputs.πθ =  OCM.ω, OCM.Λ,OCM.πθ;
    inputs.Θ̄, inputs.ρ_Θ, inputs.Σ_Θ = ones(1)*OCM.Θ̄,ones(1,1)*0.8,ones(1,1)*0.017^2;

    #Equilibrium conditions
    inputs.F = (lθ,a_,c,x,X,yᵉ)-> c==1 ? Fw(OCM,lθ,a_,x,X,yᵉ) : Fb(OCM,lθ,a_,x,X,yᵉ)
    inputs.G = (Ix,A_,X,Xᵉ,lΘ)->G(OCM,Ix,A_,X,Xᵉ,lΘ)
    inputs.f = (x⁻,x⁺)->ff(OCM,x⁻,x⁺)
    return inputs
end





# old ss
OCM_=OCModel()
τb_val=0.2
r_val=0.0416708746736356
tr_val=0.6270852509001674
assign!(OCM_,r_val,tr_val,τb_val)

# get the initial distribution for the old steady state
X̄_ = getX(OCM_)
A_0 = X̄_[6]
ω̄_0_base = sum(reshape(OCM_.ω,:,2),dims=2) #get distribution over a and θ


# # new ss
# OCM=OCModel()
# τb_val=0.25
# r_val=0.04180897741139521
# tr_val=0.6536626966299931
# assign!(OCM,r_val,tr_val,τb_val)

# new ss
OCM=OCModel()
τb_val=0.40
r_val=0.04213185316804985
tr_val=0.7230089457411855
assign!(OCM,r_val,tr_val,τb_val)

# construct the derivatives at the new steady state
inputs = construct_inputs(OCM)
ZO =ZerothOrderApproximation(inputs)
computeDerivativesF!(ZO,inputs)
computeDerivativesG!(ZO,inputs)
FO = FirstOrderApproximation(ZO,300)
compute_f_matrices!(FO)
compute_f_matrices!(FO)
compute_Lemma3!(FO)
compute_Lemma4!(FO)
compute_Corollary2!(FO)
compute_Proposition1!(FO)
compute_BB!(FO)

ω̄ = reshape(OCM.ω,:,2)
p̄ = ω̄./sum(ω̄,dims=2)
p̄[isnan.(p̄[:,1]),1] .= 1.
p̄[isnan.(p̄[:,2]),2] .= 0.
ω̄_0 = (p̄.*ω̄_0_base)[:] #apply choices under new τ_b  to initial distributon




FO.X_0 = [A_0]-ZO.P*ZO.X̄
FO.Θ_0 = [0.]
FO.Δ_0 = ω̄_0 - ZO.ω̄
solve_Xt!(FO)

Xpath = [X̄_ ZO.X̄.+FO.X̂t]


df = DataFrame(Xpath',inputs.Xlab)
df.t = 0:FO.T 

default(linewidth=2)
p1 = plot(df.t, df.A, ylabel="Capital", label="")
p2 = plot(df.t, df.Frac_b, ylabel="Fraction Self Employed", label="")
