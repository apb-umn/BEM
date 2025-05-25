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



# old ss
OCM_=OCModel()
τb_val=0.2
r_val=0.0416708746736356
tr_val=0.6270852509001674
assign!(OCM_,r_val,tr_val,τb_val)


# new ss
OCM=OCModel()
τb_val=0.25
r_val=0.04180897741139521
tr_val=0.6536626966299931
assign!(OCM,r_val,tr_val,τb_val)



ab_col_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds
ab_bor_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds
aw_bor_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds


# where do constraints bind?
bf=OCM.bf
wf=OCM.wf
@unpack alθ,Ia,Nθ,lθ, χ,agrid,  a̲, Na,Ia = OCM
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




 """
    save_policy_functions!(OCM::OCModel)

Saves the policy functions in the OCModel object
"""

function get_policy_functions(OCM::OCModel)
    @unpack bf,wf,curv_a,Na,amax,a̲,curv_h,Ia,r,σ=OCM
    #save the policy functions a,n,k,λ,v
    af(lθ,a,c) = c==1 ? wf.a[[lθ].==eachrow(OCM.lθ)][1](a) : bf.a[[lθ].==eachrow(OCM.lθ)][1](a)
    nf(lθ,a,c) = c==1 ? -exp.(lθ[2]) : bf.n[[lθ].==eachrow(OCM.lθ)][1](a)
    kf(lθ,a,c) = c==1 ? 0 : bf.k[[lθ].==eachrow(OCM.lθ)][1](a)
    yf(lθ,a,c) = c==1 ? 0 : bf.y[[lθ].==eachrow(OCM.lθ)][1](a)
    nbf(lθ,a,c) = c==1 ? 0 : bf.n[[lθ].==eachrow(OCM.lθ)][1](a)
    cf(lθ,a,c) = c==1 ? wf.c[[lθ].==eachrow(OCM.lθ)][1](a) : bf.c[[lθ].==eachrow(OCM.lθ)][1](a)
    λf(lθ,a,c) = (1+r)*cf(lθ,a,c).^(-σ)
    vf(lθ,a,c) = c==1 ? wf.v[[lθ].==eachrow(OCM.lθ)][1](a) : bf.v[[lθ].==eachrow(OCM.lθ)][1](a)
    πf(lθ,a,c) = c==1 ? 0 : bf.π[[lθ].==eachrow(OCM.lθ)][1](a)
    Ibf(lθ,a,c) = c==1 ? 0 : 1

   
    return [af,nf,kf,yf,nbf,πf,Ibf,λf,vf] #return xf
end
xf=get_policy_functions(OCM)


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

aknots,ka,aθc_sp,aθc_Ω,ℵ = get_grids(OCM)


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

#= 
using Plots
using LaTeXStrings
s=11
# Define the cutoff values
cut_bor = ab_bor_cutoff[s]
cut_col = ab_col_cutoff[s]
sel=ah.<max(cut_bor,cut_col)*1.1

# First plot: Borrowing constraint
p1 = plot(ah[sel], a′[sel, s],
    label = L"savings",
    ylabel = L"$a'$",
    title = "Borrowing Constraint",
    legend = :topleft)

# Add vertical line at cutoff
vline!(p1, [cut_bor], label = "borrowing Cutoff", color = :red, linestyle = :dash)

vline!(p1, [cut_col], label = "Collateral Cutoff", color = :blue, linestyle = :dash)
# Second plot: Collateral constraint
p2 = plot!(ah[sel], kb[sel, s],
    label = L"capital",
    xlabel = L"$a$",
    ylabel = L"$k$",
    title = "Collateral Constraint",
    legend = :topleft)

# Add vertical line at cutoff
vline!(p2, [cut_col], label = "collateral Cutoff", color = :blue, linestyle = :dash)
vline!(p2, [cut_bor], label = "borrowing Cutoff", color = :red, linestyle = :dash)

# Combine side by side
plot(p1, p2, layout = (1, 2), size = (900, 400))


 =#
function Fw(OCM::OCModel,lθ,a_,x,X,yᵉ)
    @unpack β,σ,a̲,τc,τw,γ = OCM
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
    @unpack β,σ,α_b,ν,χ,a̲,δ,τb,τc,τw,γ,α_b = OCM
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
    _,_,_,_,_,λ⁻,v⁻  = x⁻
    _,_,_,_,_,λ⁺,v⁺  = x⁺

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





# construct the derivatives
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


# get the initial distribution for the old steady state
X̄_ = getX(OCM_)
A_0 = X̄_[6]
ω̄_0_base = sum(reshape(OCM_.ω,:,2),dims=2) #get distribution over a and θ


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


using DataFrames
df = DataFrame(Xpath',inputs.Xlab)
df.t = 0:FO.T 

default(linewidth=2)
#p1 = plot(df.t, df.A, ylabel="Capital", label="")
p2 = plot(df.t, df.Frac_b, ylabel="Fraction Self Employed", label="")
