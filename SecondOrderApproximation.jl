include("FirstOrderApproximation.jl")
using TimerOutputs


####NOTE I WILL ASSUME n.a = 1 throughout this


"""
SecondOrderApproximation{Model}

Holds all the objects necessary for a second order approximation of a 
given Model.  Will assume F_x(M::Model,z) etc. exists.
"""
@with_kw mutable struct SecondOrderApproximation
    FO::FirstOrderApproximation
    T::Int = FO.T #Length of IRF
    

    #compenents of the derivative with respect to second state
    X_02::Vector{Float64} = zeros(1) #initial predetermined for second derivative
    Θ_02::Vector{Float64} = zeros(1) #initial exogenous states
    ω̂k::Matrix{Float64} =  zeros(1,1)
    ω̂ak::Matrix{Float64} =  zeros(1,1)
    x̂k::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    ŷk::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    κ̂k::Vector{Vector{Float64}} =  Vector{Vector{Float64}}(undef,1)
    X̂k::Matrix{Float64} = zeros(1,1)


    ##Lemma 2 terms
    GΘΘtk::Matrix{Float64} = zeros(1,1) #nX×T

    ##Lemma 3 terms
    xσσ::Matrix{Float64} = zeros(0,0)
    x∞::Vector{Array{Float64,3}} = Vector{Array{Float64,3}}(undef,1)
    ŷtk::Array{Float64,3} = zeros(0,0,0)
    xtk::Array{Float64,3} = zeros(0,0,0)
    Ixtkδ::Array{Float64,3} = zeros(0,0,0)
    
    #Precomputed components of xtk
    Fxk::Matrix{Matrix{Float64}} = zeros(0,0)
    FXk::Matrix{Matrix{Float64}} = zeros(0,0)
    Fyᵉk::Matrix{Matrix{Float64}} = zeros(0,0)
    FxkΘ::Vector{Matrix{Matrix{Float64}}} = zeros(0)
    FXkΘ::Vector{Matrix{Matrix{Float64}}} = zeros(0)
    FyᵉkΘ::Vector{Matrix{Matrix{Float64}}} = zeros(0)
    #delta function components
    
    ##Lemma 4 terms
    atk::Matrix{Float64} = zeros(1,1)
    btk::Matrix{Float64} = zeros(1,1) 
    ctk::Matrix{Float64} = zeros(1,1)

    #Corollary 2 terms
    Lᵃₐₐ::SparseMatrixCSC{Float64,Int64} = spzeros(1,1)
    Lᵃᵃ::SparseMatrixCSC{Float64,Int64} = spzeros(1,1)
    Lₐₐ::SparseMatrixCSC{Float64,Int64} = spzeros(1,1)

    ω̃tkᵃᵃ::Matrix{Float64} = zeros(1,1) 
    ω̃tkᵃ::Matrix{Float64} = zeros(1,1) 
    ω̃tk::Matrix{Float64} = zeros(1,1) 
    IBσσ::Matrix{Float64} = zeros(1,1)

    #Proposition 1 terms
    J∞::Array{Float64,4} = zeros(0,0,0,0)

    #Lemma 3 SV terms
    xσσΥ::Matrix{Float64} = zeros(0,0)
    #Corollary 2 SV terms
    Bσσ::Matrix{Float64} = zeros(0,0) #do we need this?
    IBσσΥ::Matrix{Float64} = zeros(1,1) #do we need this?
    IBΥ::Matrix{Float64} = zeros(1,1)



    #Outputs
    X̂tk::Matrix{Float64} = zeros(1,1)
    x̂tk::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    Ω̂tk::Matrix{Float64} = zeros(1,1) 

    X̂_ΘΘ::Matrix{Matrix{Float64}} = Matrix{Matrix{Float64}}(undef,1,1)
    x̂_ΘΘ::Matrix{Vector{Matrix{Float64}}} =  Matrix{Vector{Matrix{Float64}}}(undef,1,1)
    Ixtkδ_ΘΘ::Matrix{Array{Float64,3}} =  Matrix{Array{Float64,3}}(undef,1,1)
    
    X̂_σσ::Matrix{Float64} = zeros(1,1)
    X̂Υ::Matrix{Float64} = zeros(1,1)
    x̂_σσ::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef,1)
    Ω̄_σσ::Matrix{Float64} = zeros(1,1)

    x̂_σσΥ::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef,1)
end

function Base.copy(SO::SecondOrderApproximation)
    SOtemp = SecondOrderApproximation(FO=SO.FO)
    for k in fieldnames(SecondOrderApproximation)
        setfield!(SOtemp,k,getfield(SO,k))
    end
    return SOtemp
end



"""
    compute_Lemma2_ZZ!(SO)

Computes the GΘΘtk terms
"""
function compute_Lemma2_ZZ!(SO::SecondOrderApproximation)
    @unpack FO,ω̂k,ω̂ak,x̂k,κ̂k,X̂k,Θ_02 = SO
    @unpack ω̂t,ω̂at,x̂t,κ̂t,X̂t,Θ_0,ZO = FO
    @unpack x̄,Φ,Φₐ,ω̄,dlΓ,dG,ρ_Θ,P,Q,p,n = ZO
    T = length(x̂k)

    #Compute RFOT
    GΘΘtk = SO.GΘΘtk = zeros(n.X,T)
    IntΦ̃ = Φ * ω̄ #operator to integrate splines over ergodic
    xvec = x̄*Φ
    IntΦ̃κ =(xvec.*ω̄'.*dlΓ')*Φ' #operator to integrate κ[s] over ergodic
    for t in 1:T
        X̂t_t = X̂t[:,t]
        x̂It = x̂t[t]*IntΦ̃ + IntΦ̃κ*κ̂t[t] + FO.I*ω̂t[:,t] + FO.Ia*ω̂at[:,t]
        X̂k_t = X̂k[:,t]
        x̂Ik = x̂k[t]*IntΦ̃ + IntΦ̃κ*κ̂k[t] + FO.I*ω̂k[:,t] + FO.Ia*ω̂ak[:,t]
        if t == 1
            X̂t_t_ = FO.X_0
            X̂k_t_ = SO.X_02
        else
            X̂t_t_ = P*view(X̂t,:,t-1)
            X̂k_t_ = P*view(X̂k,:,t-1)
        end
        if t==T
            X̂t_tᵉ = zeros(n.X)
            X̂k_tᵉ = zeros(n.X)
        else
            X̂t_tᵉ = X̂t[:,t+1]
            X̂k_tᵉ = X̂k[:,t+1]
        end

        θt,θk = ρ_Θ^(t-1)*Θ_0,ρ_Θ^(t-1)*Θ_02
        GΘΘtk[:,t] = (dG.xx⋅(x̂It,x̂Ik) .+ dG.xX⋅(x̂It,X̂k_t) .+ dG.xX⋅(x̂Ik,X̂t_t)  .+ dG.xΘ⋅(x̂It,θk) .+ dG.xΘ⋅(x̂Ik,θt)
                .+ dG.xXᵉ⋅(x̂It,X̂k_tᵉ) .+ dG.xXᵉ⋅(x̂Ik,X̂t_tᵉ) .+dG.xX_⋅(x̂It,X̂k_t_) .+ dG.xX_⋅(x̂Ik,X̂t_t_)
                .+ dG.X_X_⋅(X̂t_t_,X̂k_t_) .+ dG.X_X⋅(X̂t_t_,X̂k_t) .+ dG.X_X⋅(X̂k_t_,X̂t_t) .+ dG.X_Xᵉ⋅(X̂t_t_,X̂k_tᵉ)
                .+ dG.X_Xᵉ⋅(X̂k_t_,X̂t_tᵉ) .+ dG.X_Θ⋅(X̂t_t_,θk) .+ dG.X_Θ⋅(X̂k_t_,θt) .+ dG.XX⋅(X̂t_t,X̂k_t)
                .+ dG.XXᵉ⋅(X̂t_t,X̂k_tᵉ) .+ dG.XXᵉ⋅(X̂k_t,X̂t_tᵉ) .+ dG.XΘ⋅(X̂t_t,θk) .+ dG.XΘ⋅(X̂k_t,θt)
                .+ dG.XᵉXᵉ⋅(X̂t_tᵉ,X̂k_tᵉ) .+ dG.XᵉΘ⋅(X̂t_tᵉ,θk) .+ dG.XᵉΘ⋅(X̂k_tᵉ,θt) .+ dG.ΘΘ⋅(θt,θk))

    end
    
end


"""
    compute_E_components!(SO)  
    
Computes the AF_Zx,AF_ZX,AF_Zx′ matrices that are components of the 
computation of the E matrices.
"""
function compute_lemma3_components!(SO::SecondOrderApproximation)
    @unpack FO,T = SO
    @unpack ZO,X̂t,x̂t,ŷt,ȳ_a = FO
    @unpack x̄,Φ̃ᵉₐ,Φ̃,Φ̃ᵉ,p,Q,dF,n = ZO

    Fxk = SO.Fxk = Matrix{Matrix{Float64}}(undef,n.sp,T)
    FXk = SO.FXk = Matrix{Matrix{Float64}}(undef,n.sp,T)
    Fyᵉk= SO.Fyᵉk = Matrix{Matrix{Float64}}(undef,n.sp,T)

    Eȳ_a = ȳ_a*Φ̃ᵉ
    for k in 1:T
        X̂t_k = Q*view(X̂t,:,k)
        x̂t_k = x̂t[k]*Φ̃
        ât_k = (p*x̂t_k)[:]
        if k < T
            Eŷt_k = ŷt[k+1]*Φ̃ᵉ 
        else
            Eŷt_k = zeros(n.y,n.sp)
        end
        for j in 1:n.sp
            Eȳ′t_k = @views Eȳ_a[:,j].* ât_k[j] .+ Eŷt_k[:,j]#t==T ? (x̄_a*Φ̃ᵉj)*ât_t : (x̄_a*Φ̃ᵉj)*ât_t + x̄_Z[t+1]*Φ̃ᵉj
            x̂t_kj = @views x̂t_k[:,j]
            @tensor Fxk[j,k][i,l] := dF.xx[j][i,l,o]*x̂t_kj[o] + dF.xX[j][i,l,o]*X̂t_k[o] + dF.xyᵉ[j][i,l,o]*Eȳ′t_k[o]
            @tensor FXk[j,k][i,l] := dF.Xx[j][i,l,o]*x̂t_kj[o] + dF.XX[j][i,l,o]*X̂t_k[o] + dF.Xyᵉ[j][i,l,o]*Eȳ′t_k[o]
            @tensor Fyᵉk[j,k][i,l] := dF.yᵉx[j][i,l,o]*x̂t_kj[o] + dF.yᵉX[j][i,l,o]*X̂t_k[o] + dF.yᵉyᵉ[j][i,l,o]*Eȳ′t_k[o]
        end
    end
end

"""
    compute_ŷtk!(SO)

Computes the ŷtk terms for a given k
"""
function compute_ŷtk!(SO::SecondOrderApproximation)
    @unpack FO,Fxk,FXk,Fyᵉk,X̂k,x̂k,ŷk = SO
    @unpack ZO,f,X̂t,x̂t,ŷt,ȳ_a,T = FO
    @unpack x̄,Φ̃ᵉₐ,Φ̃,Φ̃ᵉ,Φ̃ᵉₐ,p,Q,n,df,Δ⁺,Δ⁻ = ZO
    luΦ̃ = lu(Φ̃)

    ŷtk = SO.ŷtk = zeros(n.y,n.sp,T)
    ytktemp = zeros(n.y,n.sp)
    for s in 1:T
        x̂t⁺_s = x̂t[s]*Φ̃*Δ⁺
        x̂t⁻_s = x̂t[s]*Φ̃*Δ⁻
        x̂k⁺_s = x̂k[s]*Φ̃*Δ⁺
        x̂k⁻_s = x̂k[s]*Φ̃*Δ⁻

        for j in 1:n.sp
             ytktemp[:,j] .= df.x⁺x⁺[j]⋅(x̂t⁺_s[:,j],x̂k⁺_s[:,j]) .+ df.x⁺x⁻[j]⋅(x̂t⁺_s[:,j],x̂k⁻_s[:,j]) .+ df.x⁻x⁺[j]⋅(x̂t⁻_s[:,j],x̂k⁺_s[:,j]) .+ df.x⁻x⁻[j]⋅(x̂t⁻_s[:,j],x̂k⁻_s[:,j])
        end
        ŷtk[:,:,s] = ytktemp/luΦ̃
    end
end

"""
    compute_lemma3_ZZ!(SO)

Computes the xtk terms for a given k
"""
function compute_lemma3_ZZ!(SO::SecondOrderApproximation)
    @unpack FO,Fxk,FXk,Fyᵉk,X̂k,x̂k,ŷk,ŷtk = SO
    @unpack ZO,f,X̂t,x̂t,ŷt,ȳ_a = FO
    @unpack x̄,Φ̃ᵉₐ,Φ̃,Φ̃ᵉ,Φ̃ᵉₐ,p,Q,n,dF,Δ⁺,Δ⁻,df = ZO
    T = length(x̂k) #allow for x̄_Z2 to have a shorter truncation
    luΦ̃ = lu(Φ̃)

    xtk_temp = zeros(n.x,n.sp)
    ytk_temp = zeros(n.y,n.sp)
    xtk = SO.xtk = zeros(n.x,n.sp,T)
    Eȳ_aa = ȳ_a*Φ̃ᵉₐ
    Eȳ_a = ȳ_a*Φ̃ᵉ
    for s in reverse(1:T)
        X̂t_s = @views Q*X̂t[:,s]
        X̂k_s = @views Q*X̂k[:,s]
        x̂t_s = x̂t[s]*Φ̃
        ât_s = (p*x̂t_s)[:]
        x̂k_s = x̂k[s]*Φ̃
        âk_s = (p*x̂k_s)[:]
        if s < T
            Eŷt_a_s = ŷt[s+1]*Φ̃ᵉₐ
            Eŷk_a_s = ŷk[s+1]*Φ̃ᵉₐ
            Eŷk_s  =  ŷk[s+1]*Φ̃ᵉ
            Eŷtk_s = ŷtk[:,:,s+1]*Φ̃ᵉ
            Eytk_s = ytk_temp*Φ̃ᵉ
        else
            Eŷt_a_s = zeros(n.y,n.sp)
            Eŷk_a_s = zeros(n.y,n.sp)
            Eŷk_s  =  zeros(n.y,n.sp)
            Eŷtk_s = zeros(n.y,n.sp)
            Eytk_s = zeros(n.y,n.sp)
        end
        
        for j in 1:n.sp
            Eȳ′k_s = view(Eȳ_a,:,j)*âk_s[j] .+ view(Eŷk_s,:,j)

            xtk_temp[:,j]  .= dF.yᵉ[j]*(view(Eȳ_aa,:,j)*ât_s[j]*âk_s[j].+ view(Eŷk_a_s,:,j)*ât_s[j] .+ view(Eŷt_a_s,:,j)*âk_s[j])
            xtk_temp[:,j] .+= Fxk[j,s]*view(x̂k_s,:,j) .+ FXk[j,s]*X̂k_s .+ Fyᵉk[j,s]*Eȳ′k_s
            xtk_temp[:,j] .+= dF.yᵉ[j]*(view(Eytk_s,:,j)+view(Eŷtk_s,:,j))
            xtk_temp[:,j] = f[j]*view(xtk_temp,:,j)
        end
        xtk[:,:,s] = xtk_temp/luΦ̃  

        #compute effect on y terms
        xtk_temp⁺ = xtk_temp*Δ⁺
        xtk_temp⁻ = xtk_temp*Δ⁻
        for j in 1:n.sp
            ytk_temp[:,j] .= df.x⁺[j]*xtk_temp⁺[:,j] .+ df.x⁻[j]*xtk_temp⁻[:,j]
        end
        ytk_temp = ytk_temp[:,:]/luΦ̃
    end
end


function compute_lemma3_ZZ_kink!(SO)
    @unpack FO,x̂k = SO
    @unpack ZO,x̂t,x̄_a = FO
    @unpack x̄,Φ̃,p,Q,n,ℵ,aθc_sp = ZO
    nℵ = length(ℵ)
    T = length(x̂k)
    SO.Ixtkδ =  zeros(n.x,n.sp,T)
    #TODO: FIX THIS
    return
    
    
    Φ̃m = Φ̃[:,ℵ]
    Φ̃p = Φ̃[:,ℵ.+1]
    x̄Δ_a = x̄_a*Φ̃[:,ℵ.+1].-x̄_a*Φ̃[:,ℵ]
    āΔ_a = p*x̄Δ_a
    xtkδ  = zeros(n.x,nℵ,T)
    for t in 1:T
        x̄Δ_Z1 = x̂t[t]*Φ̃p .- x̂t[t]*Φ̃m
        x̄Δ_Z2 = x̂k[t]*Φ̃p .- x̂k[t]*Φ̃m
        astar_Z1 = -(āΔ_a).^(-1).*(p*x̄Δ_Z1)
        astar_Z2 = -(āΔ_a).^(-1).*(p*x̄Δ_Z2)
        xtkδ[:,:,t] .=  x̄Δ_a.*astar_Z1.*astar_Z2
    end
    #now construct Ixtkδ
    Ixtkδ = zeros(n.x,n.sp,T)
    for (iℵ,j) in enumerate(ℵ)
        #mask_stepfunction = filter(i->ẑ[i][1]>ẑ[j][1] && ẑ[i][2:end]==ẑ[j][2:end],1:n.ẑ) #mask for step function
        mask_stepfunction = filter(i->aθ_sp[i,1]>aθ_sp[j,1] && aθ_sp[i,2:end]==aθ_sp[j,2:end],1:n.sp) #mask for step function
        Ixtkδ[:,mask_stepfunction,:] .+= reshape(xtkδ[:,iℵ,:],n.x,1,T)
    end
    SO.Ixtkδ =  zeros(n.x,n.sp,T)
    luΦ̃ = lu(Φ̃)
    for t in 1:T
        @views SO.Ixtkδ[:,:,t] .= (luΦ̃'\Ixtkδ[:,:,t]')'
    end
end


"""
    compute_Lemma4_ZZ!(SO)

Computes the a and b elements from Lemma 4
"""
function compute_Lemma4_ZZ!(SO)
    @unpack FO,ω̂k,ω̂ak,x̂k,X̂k,κ̂k,xtk,Ixtkδ = SO
    @unpack ZO,ω̂t,ω̂at,x̂t,κ̂t,X̂t,L,x̄_a = FO
    @unpack n,Φ,Φₐ,p,pκ,Δ,ω̄,Λ,Q,x̄,dlΓ,d2lΓ = ZO
    #Construct kink terms

    T = length(x̂k)
    atk= SO.atk = zeros(n.Ω,T)
    btk= SO.btk = zeros(n.Ω,T)
    ctk= SO.ctk = zeros(n.Ω,T)
    a_a = ((p*x̄_a)*Φ)[:]
    κ_a = ((pκ*x̄_a*Δ)*Φ)[:]
    p_a = dlΓ.*κ_a
    for t in 2:T 
        #TODO:CHECK TIMING!
        ât = ((p*x̂t[t-1])*Φ)[:]
        κ̂t_t = (κ̂t[t-1]'*Φ)[:]
        κ̂t_t_a = (κ̂t[t-1]'*Φₐ)[:]
        âk = ((p*x̂k[t-1])*Φ)[:]
        κ̂k_t = (κ̂k[t-1]'*Φ)[:]
        κ̂k_t_a = (κ̂k[t-1]'*Φₐ)[:]
        ât_a = ((p*x̂t[t-1])*Φₐ)[:]
        âk_a = ((p*x̂k[t-1])*Φₐ)[:]

        atk_t = ((p*xtk[:,:,t-1])*Φ)[:] .+ ((p*Ixtkδ[:,:,t-1])*Φₐ)[:]
        κtk_t = ((pκ*xtk[:,:,t-1])*Δ*Φ)[:] .+ ((pκ*Ixtkδ[:,:,t-1])*Δ*Φₐ)[:]
        

        p̂t_t = dlΓ.*κ̂t_t
        p̂k_t = dlΓ.*κ̂k_t
        p̂tk_t = dlΓ.*κtk_t .+ d2lΓ.*κ̂t_t.*κ̂k_t
        p̂t_t_a = dlΓ.*κ̂t_t_a .+ d2lΓ.*κ̂t_t.*κ_a
        p̂k_t_a = dlΓ.*κ̂k_t_a .+ d2lΓ.*κ̂k_t.*κ_a

    
        #âk = ((p*x̂k[t-1])*Φ̃ā)[:]
        @views atk[:,t] = Λ*(a_a.*ω̂at[:,t-1].*âk .+ a_a.*ω̂ak[:,t-1].*ât .+ ât.*âk.*ω̄)  ##(aa) level
        @views btk[:,t] .= Λ*( 
            (ât.*p̂k_t .+ âk.*p̂t_t).*ω̄ .+ âk.*ω̂t[:,t-1] .+ ât.*ω̂k[:,t-1] .+
            (p_a.*âk .+ a_a.*p̂k_t .+ âk_a).*ω̂at[:,t-1] .+
            (p_a.*ât .+ a_a.*p̂t_t .+ ât_a).*ω̂ak[:,t-1] .+
            (atk_t).*ω̄ 
        )

        @views ctk[:,t] = Λ*(
            p̂tk_t.*ω̄  .+ p̂k_t.*ω̂t[:,t-1] .+ p̂t_t.*ω̂k[:,t-1] .+
            p̂t_t_a.*ω̂ak[:,t-1] .+ p̂k_t_a.*ω̂at[:,t-1]
        )

    end
end

function construct_Laa!(SO)
    @unpack FO = SO
    @unpack ZO,x̂t,X̂t,x̄_a = FO
    @unpack Φ,Φₐ,p,pκ,Δ,ω̄,Λ,Q,x̄,n,dlΓ,d2lΓ = ZO

    #ā_a = ((p*x̄)*Φₐ)[:]  #1xI array should work with broadcasting
    ā_a = clamp.(((p*x̄_a)*Φ)[:],-2,2)  #1xI array should work with broadcasting
    #ā_a = ((p*x̄_a)*Φ)[:]  #1xI array should work with broadcasting
    ā_aa = ((p*x̄_a)*Φₐ)[:]#((p*x̄_zz)*Φ)[:]#  #1xI array should work with broadcasting
    κ_a = ((pκ*x̄_a*Δ)*Φ)[:]
    κ_aa = ((pκ*x̄_a*Δ)*Φₐ)[:]
    p_a = dlΓ.*κ_a
    p_aa = d2lΓ.*κ_a.*κ_a .+ dlΓ.*κ_aa
    #This code stops the creation of a large dense matrix
    Lᵃᵃ = SO.Lᵃᵃ =copy(Λ)
    Lᵃₐₐ = SO.Lᵃₐₐ =copy(Λ)
    Lₐₐ = SO.Lₐₐ =copy(Λ)
    for j in 1:n.Ω
        for i in nzrange(Lᵃₐₐ,j)
            @inbounds Lᵃₐₐ.nzval[i] *= ā_aa[j] + 2*ā_a[j]*p_a[j]
        end
        for i in nzrange(Lᵃᵃ,j)
            @inbounds Lᵃᵃ.nzval[i] *= ā_a[j]*ā_a[j] 
        end
        for i in nzrange(Lₐₐ,j)
            @inbounds Lₐₐ.nzval[i] *= p_aa[j]
        end
    end
end


function compute_Corollary2_ZZ!(SO)
    @unpack FO,Lᵃᵃ,Lᵃₐₐ,Lₐₐ,atk,btk,ctk = SO
    @unpack ZO,La,Ma,L,M = FO
    @unpack ω̄,p,n,Φₐ,Λ = ZO
    #apply iteration
    #Iz = length(ω̄)
    T = size(btk)[2] #use possibly shorter horizon
    ω̃tkᵃᵃ = SO.ω̃tkᵃᵃ = zeros(n.Ω,T)
    ω̃tkᵃ = SO.ω̃tkᵃ = zeros(n.Ω,T)
    ω̃tk = SO.ω̃tk = zeros(n.Ω,T)

    for t in 2:T
        @views ω̃tkᵃᵃ[:,t] .= Lᵃᵃ * ω̃tkᵃᵃ[:,t-1] .+ atk[:,t]
        @views ω̃tkᵃ[:,t]  .= La * ω̃tkᵃ[:,t-1] .+ Lᵃₐₐ * ω̃tkᵃᵃ[:,t-1] .+ btk[:,t]
        @views ω̃tk[:,t]   .= Λ * ω̃tk[:,t-1] .+ L * ω̃tkᵃ[:,t-1] .+ Lₐₐ *ω̃tkᵃᵃ[:,t-1]  .+ ctk[:,t]
    end
end


function compute_Intxtk(SO)
    @unpack FO,ω̂k,ω̂ak,x̂k,X̂k,κ̂k,xtk,Ixtkδ,ω̃tkᵃᵃ,ω̃tkᵃ,ω̃tk = SO
    @unpack ZO,ω̂t,ω̂at,x̂t,κ̂t,X̂t,L,x̄_a,T = FO
    @unpack n,Φ,Φₐ,p,pκ,Δ,ω̄,Λ,Q,x̄,dlΓ,d2lΓ = ZO
    
    Intxtk = zeros(n.x,T)
    κ_a = ((pκ*x̄_a*Δ)*Φ)[:]
    κ_aa = ((pκ*x̄_a*Δ)*Φₐ)[:]
    p_a = dlΓ.*κ_a
    p_aa = d2lΓ.*κ_a.*κ_a .+ dlΓ.*κ_aa
    xΦ = x̄*Φ
    x_a = x̄_a*Φ
    x_aa = x̄_a*Φₐ

    for t in 1:T
        x̂t_t = x̂t[t]*Φ
        κ̂t_t = (κ̂t[t]'*Φ)[:]
        κ̂t_t_a = (κ̂t[t]'*Φₐ)[:]
        x̂k_t = x̂k[t]*Φ
        κ̂k_t = (κ̂k[t]'*Φ)[:]
        κ̂k_t_a = (κ̂k[t]'*Φₐ)[:]
        x̂t_t_a = x̂t[t]*Φₐ
        x̂k_t_a = x̂k[t]*Φₐ

        xtk_t = xtk[:,:,t]*Φ .+ Ixtkδ[:,:,t]*Φₐ
        κtk_t = ((pκ*xtk[:,:,t])*Δ*Φ)[:] .+ ((pκ*Ixtkδ[:,:,t])*Δ*Φₐ)[:]

        p̂t_t = dlΓ.*κ̂t_t
        p̂k_t = dlΓ.*κ̂k_t
        p̂tk_t = dlΓ.*κtk_t .+ d2lΓ.*κ̂t_t.*κ̂k_t
        p̂t_t_a = dlΓ.*κ̂t_t_a .+ d2lΓ.*κ̂t_t.*κ_a
        p̂k_t_a = dlΓ.*κ̂k_t_a .+ d2lΓ.*κ̂k_t.*κ_a


        @views Intxtk[:,t] = (
            (xtk_t .+ x̂t_t.*p̂k_t' .+ x̂k_t.*p̂t_t' .+ xΦ.*p̂tk_t')*ω̄ .+
            (x̂t_t .+ xΦ.*p̂t_t')*ω̂k[:,t] .+
            (x̂t_t_a .+ x̂t_t.*p_a' .+ x_a.*p̂t_t' .+ xΦ.*p̂t_t_a')*ω̂ak[:,t] .+
            (x̂k_t .+ xΦ.*p̂k_t')*ω̂t[:,t] .+
            (x̂k_t_a .+ x̂k_t.*p_a' .+ x_a.*p̂k_t' .+ xΦ.*p̂k_t_a')*ω̂at[:,t] .+
            FO.I*ω̃tk[:,t] .+ FO.Ia*ω̃tkᵃ[:,t] .+
            (x_aa .+ 2*x_a.*p_a' .+ p_aa')*ω̃tkᵃᵃ[:,t]
        )
    end
    return Intxtk
end

"""
    compute_XZZ!(SO)

Computes the path of the second order derivatives X_ZZ
"""
function compute_XZZ!(SO::SecondOrderApproximation)
    @unpack FO,x̂k,GΘΘtk,xtk,Ixtkδ = SO
    @unpack ZO,X̂t,luBB,x̄_a = FO
    @unpack x̄,Φ,Φₐ,ω̄,dG,P,Q,p,n = ZO
    T = size(FO.X̂t,2)
    T2 = length(x̂k)
    Intxtk = compute_Intxtk(SO)
    #Compute RFOT
    AA = zeros(n.X,T)
    for t in 1:T
        @views AA[:,t] .= dG.x*Intxtk[:,t] .+ GΘΘtk[:,t] #Htk+GΘΘtk
    end

    #SO.X̂tk = reshape(-BB[1:n.X*T,1:n.X*T]\AA[:],n.X,T)
    SO.X̂tk = reshape(-(luBB\AA[:])[1:n.X*T],n.X,T)
end

"""
    compute_xZZ!(SO)

Computes the path of the second order derivatives x_ZZ
"""
function compute_xZZ!(SO::SecondOrderApproximation)
    @unpack FO,xtk,X̂tk = SO
    @unpack ZO,T = FO
    @unpack p,Q,n = ZO

    x̂tk = SO.x̂tk # =Vector{Matrix{Float64}}(undef,T)
#    x = permutedims(FO.x,[1,3,2])
    flipdims = xs->permutedims(xs,[1,3,2])
    x = flipdims.(FO.x)
    for t in 1:T
        x̂tk[t] = xtk[:,:,t]
        for s in 1:T-t+1
            @views x̂tk[t] += x[s]*(Q*X̂tk[:,t+s-1])
        end
    end
end

"""
    compute_xZZ!(SO)

Computes the path of the second order derivatives x_ZZ
"""
function compute_xZZ0!(SO::SecondOrderApproximation)
    @unpack FO,xtk,X̂tk = SO
    @unpack ZO,T = FO
    @unpack p,Q,n = ZO

    x̂tk = SO.x̂tk = Vector{Matrix{Float64}}(undef,T)
    flipdims = xs->permutedims(xs,[1,3,2])
    x = flipdims.(FO.x)
    x̂tk[1] = xtk[:,:,1]
    for s in 1:T
        @views x̂tk[1] += x[s]*(Q*X̂tk[:,s])
    end
end


"""
    compute_lemma3_σσ!(SO)

Computes the x_σσ terms from Lemma 3, as well as x∞ terms.
"""
function compute_lemma3_σσ!(SO::SecondOrderApproximation)
    @unpack FO,T,x̂_ΘΘ,x∞,Ixtkδ_ΘΘ = SO
    @unpack ZO,f,x̂_Θt = FO
    @unpack x̄,Φ̃,Φ̃ᵉ,Φ̃ᵉₐ,p,dF,n,Σ_Θ = ZO
    B = spzeros(n.x*n.sp,n.x*n.sp)
    F̂_σσ = zeros(n.x,n.sp)
    fF_X = zeros(n.x,n.sp,n.Q)
    fF_x′ = Vector{Matrix{Float64}}(undef,n.sp)

    Ex̂_ΘΘ′ = sum((x̂_ΘΘ[i][1]*Φ̃ᵉ.+Ixtkδ_ΘΘ[i][:,:,1]*Φ̃ᵉₐ)*Σ_Θ[i] for i in 1:length(Σ_Θ))
    #Ex̄_σσ′ = sum(x̄_Θ[i][1]*Φ̃ᵉ*μ_Θσσ[i] for i in 1:length(μ_Θσσ)) #account for change in mean
    for j in 1:n.sp

        fF_x′[j] = f[j]*dF.x′[j]
        B[1+(j-1)*n.x:j*n.x,:] = kron(sparse(Φ̃[:,j]'),Matrix(I,n.x,n.x)) - kron(sparse(Φ̃ᵉ[:,j]'),fF_x′[j])

        F̂_σσ[:,j] = fF_x′[j]*(Ex̂_ΘΘ′[:,j])#+Ex̂_σσ′[:,j])
        fF_X[:,j,:] = f[j] * dF.X[j]
    end
    luB = lu(B)
    xσσ = SO.xσσ = reshape(luB\F̂_σσ[:],n.x,:)
    x̂_σσX = reshape(luB\reshape(fF_X,:,n.Q),n.x,n.sp,n.Q)
    x∞ = SO.x∞ = Vector{Array{Float64,3}}(undef,T+1)
    x∞[1] = permutedims(x̂_σσX,[1,3,2])
    x∞temp = zeros(n.x,n.Q,n.sp)
    luΦ̃ = lu(Φ̃)
    for s in 2:T+1
        Ex∞ = x∞[s-1]*Φ̃ᵉ
        for j in 1:n.sp
            @views x∞temp[:,:,j] .= fF_x′[j]*Ex∞[:,:,j]
        end
        x∞[s] = x∞temp/luΦ̃
    end
end



function compute_corollary2_σσ!(SO::SecondOrderApproximation)
    @unpack FO,xσσ = SO
    @unpack ZO,L,M,T,ILM = FO
    @unpack p,n = ZO
    aσσ = (p*xσσ)[:]
    SO.IBσσ = cumsum([zeros(n.x) ILM*aσσ][:,1:T],dims=2)
end


function compute_proposition1_σσ!(SO::SecondOrderApproximation) 
    @unpack FO,x∞,xσσ,X̂_ΘΘ = SO
    @unpack ZO,J,T,L,M = FO
    @unpack p,ω̄,Φ,x̄,Φₐ,dG,n,Q,P  = ZO
    Iop = x̄*Φₐ
    IntΦ̃ = Φ * ω̄
    J∞ = SO.J∞ = copy(J)
    A∞ = zeros(n.Ω,n.Q)
    for t in 1:T
        a∞ = (p*x∞[T-t+1])[1,:,:]'
        @views J∞[:,t,:,T] .= x∞[T-t+1]*IntΦ̃+Iop*A∞#ILM[:,T-t+1,:]*a∞
        A∞ = L*A∞+M*a∞
    end

    ITT = sparse(I,T,T)
    ITT_ = spdiagm(-1=>ones(T-1))
    ITTᵉ = spdiagm(1=>ones(T-1))
    ITTᵉ[T,T] = 1 #last

    BBσσ = kron(ITT,dG.x)*reshape(J∞,n.x*T,:)*kron(ITT,Q) .+ kron(ITT,dG.X) .+ kron(ITT_,dG.X_*P) .+ kron(ITTᵉ,dG.Xᵉ) 
    
    EX_ΘΘᵉ = sum(X̂_ΘΘ[i][:,1]*Σ_Θ[i] for i in 1:length(Σ_Θ)) 

    AAσσ = -dG.x*(xσσ*IntΦ̃ .+ SO.IBσσ) .+ (dG.Xᵉ*EX_ΘΘᵉ)[:]

    SO.X̂_σσ = reshape(BBσσ\AAσσ[:],n.X,:)
end


function compute_x̄σσ!(SO::SecondOrderApproximation)
    @unpack FO,xσσ,X̂_σσ = SO
    @unpack ZO,T = FO
    @unpack p,Q,n = ZO
    x̂_σσ = SO.x̂_σσ =Vector{Matrix{Float64}}(undef,T)
    #    x = permutedims(FO.x,[1,3,2])
    flipdims = xs->permutedims(xs,[1,3,2])
    x = flipdims.(FO.x)
    x∞ = flipdims.(SO.x∞)
    QX_σσ = Q*X̂_σσ
    for t in 1:T
        x̂_σσ[t] = copy(xσσ)
        for s in 1:T-t
            @views x̂_σσ[t] += x[s]*QX_σσ[:,t+s-1]
        end
        @views x̂_σσ[t] += x∞[T-t+1]*QX_σσ[:,T]
    end
end


"""
    compute_lemma3_SV!(SO)

Computes the xσσΥ terms from Lemma 3 SV 
"""
function compute_lemma3_SV!(SO::SecondOrderApproximation)
    @unpack FO,T,x̂_ΘΘ,x∞ = SO
    @unpack ZO,f,x̂_Θt = FO
    @unpack Φ̃,Φ̃ᵉ,Σ_ΘΥ,ρΥ,dF,n = ZO
    B = spzeros(n.x*n.sp,n.x*n.sp)
    F̂_σσ = zeros(n.x,n.sp)
    fF_x′ = Vector{Matrix{Float64}}(undef,n.sp)

    Ex̄_ΘΘ′Υ = sum((x̂_ΘΘ[i][1]*Φ̃ᵉ)*Σ_ΘΥ[i] for i in 1:length(Σ_ΘΥ))
    for j in 1:n.sp

        fF_x′[j] = f[j]*dF.x′[j]
        B[1+(j-1)*n.x:j*n.x,:] = kron(sparse(Φ̃[:,j]'),Matrix(I,n.x,n.x)) - ρΥ*kron(sparse(Φ̃ᵉ[:,j]'),fF_x′[j])

        F̂_σσ[:,j] = fF_x′[j]*Ex̄_ΘΘ′Υ[:,j]
    end
    luB = lu(B)
    SO.xσσΥ = reshape(luB\F̂_σσ[:],n.x,:)
end


function compute_corollary2_SV(SO::SecondOrderApproximation)
    @unpack FO,xσσΥ = SO
    @unpack ZO,L,T,ILM = FO
    @unpack p,ω̄,Φ,Λ,ρΥ,n = ZO
    IntΦ̃ = Φ*ω̄
    
    ILMpxσσΥ = ILM*(p*xσσΥ)[:] #n.x×T
    IBΥ =  zeros(n.x,T)
    IBΥ[:,1] = xσσΥ*IntΦ̃
    for t in 2:T
        @views IBΥ[:,t] = ρΥ.*IBΥ[:,t-1] .+ ILMpxσσΥ[:,t-1] 
    end
    SO.IBΥ = IBΥ
end


function compute_proposition1_SV(SO::SecondOrderApproximation)
    @unpack FO,IBΥ,xσσΥ,X̂_ΘΘ = SO
    @unpack ZO,BB,T = FO 
    @unpack dG, n,Σ_ΘΥ,ρΥ = ZO
    ρΥt = ρΥ.^(0:T-1)
    EX̂_ΘΘᵉΥ = sum(X̂_ΘΘ[i][:,1]*Σ_ΘΥ[i] for i in 1:length(Σ_ΘΥ))
    AA = dG.x*IBΥ .+ (dG.Xᵉ*EX̂_ΘΘᵉΥ).*ρΥt'

    SO.X̂Υ = reshape(-BB\AA[:],n.X,T)
end


function compute_x̄σσΥ!(SO::SecondOrderApproximation)
    @unpack FO,xσσΥ,X̂Υ = SO
    @unpack ZO,T = FO
    @unpack p,Q,n = ZO
    x̂_σσΥ = SO.x̂_σσΥ =Vector{Matrix{Float64}}(undef,T)
    flipdims = xs->permutedims(xs,[1,3,2])
    x = flipdims.(FO.x)
    QX_σσΥ = Q*X̂Υ
    for t in 1:T
        x̂_σσΥ[t] = copy(xσσΥ)
        for s in 1:T-t+1
            @views x̂_σσΥ[t] += x[s]*QX_σσΥ[:,t+s-1]
        end
    end
end


"""
    computeSecondOrder!(SO)

Computes the second order derivatives with 
"""
function computeSecondOrder!(SO::SecondOrderApproximation)
    @unpack FO = SO
    @unpack ZO = FO
    @unpack n = ZO

    SO.X̂_ΘΘ = Matrix{Matrix{Float64}}(undef,n.Θ,n.Θ)
    SO.x̂_ΘΘ =  Matrix{Vector{Matrix{Float64}}}(undef,n.Θ,n.Θ)
    SO.Ixtkδ_ΘΘ = Matrix{Array{Float64,3}}(undef,n.Θ,n.Θ)
    ## initialization of aggregate state at 0
    #SO.X_02 = FO.X_0
    SO.X_02 = zeros(n.A) #initial predetermined for second derivative

    # initialization for compute_XZZjk
    SO.FxkΘ = Vector{Matrix{Matrix{Float64}}}(undef,n.Θ)
    SO.FXkΘ = Vector{Matrix{Matrix{Float64}}}(undef,n.Θ)
    SO.Fx′kΘ   = Vector{Matrix{Matrix{Float64}}}(undef,n.Θ)


    for i in 1:n.Θ
        # initialization direction of the first order shock
        FO.Θ_0 = I[1:n.Θ,i]
        FO.x̂t = FO.x̂_Θt[i]
        FO.X̂t = FO.X̂_Θt[i]
        FO.Ω̂t = FO.Ω̂_Θt[i]
        compute_lemma3_components!(SO)
        SO.FxkΘ[i]= SO.Fxk
        SO.FXkΘ[i]= SO.FXk
        SO.Fx′kΘ[i]= SO.Fx′k

        for j in 1:n.Θ
            #setup second direction
            SO.Θ_02 = I[1:n.Θ,j] #initial exogenous states
            SO.x̂k = FO.x̂_Θt[j]
            SO.X̂k = FO.X̂_Θt[j]
            SO.Ω̂k = FO.Ω̂_Θt[j]
            compute_lemma3_ZZ_kink!(SO)
            compute_Lemma2_ZZ!(SO)
            compute_lemma3_ZZ!(SO)
            compute_Lemma4_ZZ!(SO)
            construct_Laa!(SO)
            compute_Corollary2_ZZ!(SO)
            compute_XZZ!(SO)
            compute_xZZ0!(SO)
            #compute_xZZ!(SO) only need the initial period x_ZZ for σσ
            SO.X̂_ΘΘ[i,j] = SO.X̂tk
            SO.X̂_ΘΘ[j,i] = SO.X̂tk
            SO.x̂_ΘΘ[i,j] = SO.x̂tk
            SO.x̂_ΘΘ[j,i] = SO.x̂tk
            SO.Ixtkδ_ΘΘ[i,j] = SO.Ixtkδ
            SO.Ixtkδ_ΘΘ[j,i] = SO.Ixtkδ
        end 
    end 
    compute_lemma3_σσ!(SO)
    compute_corollary2_σσ!(SO)
    compute_proposition1_σσ!(SO)
end 


#TODO: rename based on k
function compute_XZZjk(SO::SecondOrderApproximation,iΘ,jΘ,Tjs)  #i.e. ks
    @unpack FO = SO
    @unpack ZO, T = FO
    @unpack P, n = ZO
    

    SO.Fxk = SO.FxkΘ[iΘ]
    SO.FXk = SO.FXkΘ[iΘ] 
    SO.Fx′k= SO.Fx′kΘ[iΘ]
    X̂tk_ijtemp = Vector{Matrix{Float64}}(undef,maximum(Tjs))
    Threads.@threads for j in Tjs
        SOtemp = copy(SO)
        SOtemp.x̂k = FO.x̂_Θt[jΘ][j+1:end]
        SOtemp.X̂k = FO.X̂_Θt[jΘ][:,j+1:end]
        SOtemp.X_02 = P*FO.X̂_Θt[jΘ][:,j]
        SOtemp.Θ_02 = ρ_Θ^j*I[1:n.Θ,jΘ]
        SOtemp.Ω̂k = FO.Ω̂_Θt[jΘ][:,j+1:end]
        compute_Lemma2_ZZ!(SOtemp)
        compute_lemma3_ZZ_kink!(SOtemp)
        compute_lemma3_ZZ!(SOtemp)
        compute_Lemma4_ZZ!(SOtemp)
        compute_Corollary2_ZZ!(SOtemp)
        compute_XZZ!(SOtemp)
        X̂tk_ijtemp[j] = SOtemp.X̂tk
    end 
    X̂tk_ij = Dict()
    for j in Tjs
        X̂tk_ij[j] = X̂tk_ijtemp[j]
    end
    return X̂tk_ij
end



"""
    getPath(SO,path,iX,X_ΘΘjk)

Computes the response to a path of shocks for both a first and
second order approximation.  Uses the interaction derivatives
X_ΘΘjk.
"""
function getPath(SO::SecondOrderApproximation,path,iX,X_ΘΘjk)
    FO,ZO = SO.FO,SO.FO.ZO
    Σ_Θ = ZO.Σ_Θ

    Tsim = length(path)
    Xt=zeros(Tsim)
    X_σσt=zeros(Tsim)
    X_σσt= SO.X̂_σσ[iX,:]
    Xzerotht=ZO.X̄[iX]*ones(Tsim)
    Xfirstordert=zeros(Tsim)
    Xinteractiont=zeros(Tsim)
    Xsigmasigmat=X_σσt[1:Tsim]

    for t in 1:Tsim
        fo=0
        so=0
        # Compute the first order response 
        for s=1:t
            fo += path[s]*FO.X̂t[iX,t-s+1]
        end

        # Compute the second order response 
        for s=1:t
            for m=1:t
                lag=max(s,m)-min(s,m)
                so += path[s]*path[m]*X_ΘΘjk[lag][iX,t-max(s,m)+1]
            end
        end

        Xfirstordert[t] = fo*Σ_Θ[1,1]^0.5
        Xinteractiont[t] = so*Σ_Θ[1,1]
    end

    # Save the IRF decomposition in a DataFrame 
    Xt=DataFrame()
    Xt.zeroth=Xzerotht
    Xt.firstorder=Xfirstordert
    Xt.interaction=Xinteractiont
    Xt.risk=Xsigmasigmat
    Xt.path=path
    Xt.X=Xzerotht+Xfirstordert+0.5*Xinteractiont+0.5*Xsigmasigmat
    Xt.FO=Xzerotht+Xfirstordert
    return Xt
end
