using Parameters,BasisMatrices,SparseArrays,SuiteSparse,TensorOperations
BLAS.set_num_threads(1)
include("utilities.jl")
include("ZerothOrderApproximation.jl")

#Some Helper functions
import Base./,Base.*,Base.\
"""
    /(A::Array{Float64,3},B::SparseMatrixCSC{Float64,Int64})

Apply the inverse to the last dimension of a 3 dimensional array
"""
function /(A::Array{Float64,3},B::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64})
    ret = similar(A)
    n = size(ret,1)
    for i in 1:n
        ret[i,:,:] .= (B'\view(A,i,:,:)')' 
        #ret[i,:,:] .= A[i,:,:]/B
    end
    return ret
end

function /(A::Array{Float64,2},B::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64})
    return (B'\A')'
end


"""
    /(A::Array{Float64,3},B::SparseMatrixCSC{Float64,Int64})

Apply the inverse to the last dimension of a 3 dimensional array
"""
function \(A::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64},B::Array{Float64,3})
    n = size(A,2)
    sizeB = size(B)
    return reshape(A\reshape(B,n,:),sizeB)
end



#function *(A::Matrix{Float64},B::Array{Float64,3})
#    return @tensor C[i,k,l] := A[i,j]*B[j,k,l] 
#end

function *(A::Array{Float64,3},B::SparseMatrixCSC{Float64, Int64})
    k,m,n = size(A)
    return reshape(reshape(A,:,n)*B ,k,m,:) 
end

#function *(A::Matrix{Float64},B::Array{Float64,3})
#    return @tensor C[i,k,l] := A[i,j]*B[j,k,l] 
#end

function *(A::SparseMatrixCSC{T, Int64},B::Array{Float64,3}) where {T<:Real}
    k,m,n = size(B)
    return reshape(A*reshape(B,k,:),:,m,n) 
end

function *(A::Adjoint{T,SparseMatrixCSC{T, Int64}},B::Array{Float64,3}) where {T<:Real}
    k,m,n = size(B)
    return reshape(A*reshape(B,k,:),:,m,n) 
end


"""
FirstOrderApproximation{Model}

Holds all the objects necessary for a first order approximation of a 
given Model.  Will assume that derivatives (e.g. F_x, F_X etc.) exists.
"""
@with_kw mutable struct FirstOrderApproximation
    #M::Model #holds objects that we care about like H
    ZO::ZerothOrderApproximation
    T::Int #Length of IRF

    #Derivative direction
    Δ_0::Vector{Float64} = zeros(1) #Distribution direction (will assume 0 for now)
    X_0::Vector{Float64} = zeros(1)
    Θ_0::Vector{Float64} = zeros(1) #Θ direction

    #x̄_a
    x̄_a::Matrix{Float64} = zeros(0,0)
    #ȳ_a (mostly stored for second order approximation)
    ȳ_a::Matrix{Float64} = zeros(0,0)
    
    #Terms for Lemma 2
    f::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef,1)
    x::Vector{Array{Float64,3}} = Vector{Array{Float64,3}}(undef,1)
    
    #Terms for Lemma 4
    a::Array{Float64,3} = zeros(1,1,1) #a terms from paper
    κ::Array{Float64,3} = zeros(1,1,1) #κ terms from paper
    L::SparseMatrixCSC{Float64,Int64} = spzeros(1,1) #A operator
    La::SparseMatrixCSC{Float64,Int64} = spzeros(1,1) #A operator
    M::SparseMatrixCSC{Float64, Int64} = spzeros(1,1) #M operator
    Ma::SparseMatrixCSC{Float64, Int64} = spzeros(1,1) #M operator

    
    #Terms for Corollary 2
    I::Matrix{Float64} = zeros(1,1)   #I operator
    Ia::Matrix{Float64} = zeros(1,1)  #I operator
    IL::Array{Float64, 3} = zeros(1,1,1)
    ILM::Array{Float64, 3} = zeros(1,1,1)
    IΛ::Array{Float64, 3} = zeros(1,1,1)
    IΛM::Array{Float64, 3} = zeros(1,1,1)
    E::Array{Float64,3} = zeros(1,1,1) #expectations operators
    J::Array{Float64,4} = zeros(1,1,1,1)

    #Terms for Proposition 1
    BB::SparseMatrixCSC{Float64, Int64} = spzeros(1,1)
    luBB::SparseArrays.UMFPACK.UmfpackLU{Float64, Int64} = lu(sprand(1,1,1.))


    #Outputs
    ω̂t::Matrix{Float64} =  zeros(1,1)
    ω̂at::Matrix{Float64} =  zeros(1,1)
    x̂t::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    ŷt::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    κ̂t::Vector{Vector{Float64}} =  Vector{Vector{Float64}}(undef,1)
    X̂t::Matrix{Float64} = zeros(1,1)

end



"""
    FirstOrderApproximation(ZO::ZerothOrderApproximation, T)

Constructs a first-order approximation object based on a given zeroth-order approximation object.

# Arguments
- `ZO::ZerothOrderApproximation`: The zeroth-order approximation object.
- `T`: The number of time periods for truncation

# Returns
- `approx`: The first-order approximation object.
"""
function FirstOrderApproximation(ZO::ZerothOrderApproximation, T)
    @unpack n = ZO
    approx = FirstOrderApproximation(ZO = ZO, T = T)
    approx.f = Vector{Matrix{Float64}}(undef, n.sp)
    approx.x = Vector{Array{Float64, 3}}(undef, T)
    approx.J = zeros(ZO.n.x, T, ZO.n.Q, T)
    return approx
end


"""
    compute_f_matrices!(FO::FirstOrderApproximation)

Compute the f matrices for the FirstOrderApproximation object `FO`.
These are the building blocks of the x matrices.

# Arguments
- `FO::FirstOrderApproximation`: The FirstOrderApproximation object for which to compute the f matrices.

"""
function compute_f_matrices!(FO::FirstOrderApproximation)
    @unpack ZO,f, = FO
    @unpack x̄,Φ̃ᵉ,p,dF,df,n,Φ̃,Δ⁺,Δ⁻,Φ̃ₐ = ZO
    luΦ̃ = lu(Φ̃) #precompute inverse of basis matrix
    x̄⁺_a = reshape(x̄*Φ̃ₐ*Δ⁺,:,n.a,n.sp)
    x̄⁻_a = reshape(x̄*Φ̃ₐ*Δ⁻,:,n.a,n.sp)
    ȳ_a = zeros(n.y,n.a,n.sp)
    for j in 1:n.sp
        @views ȳ_a[:,:,j] = df.x⁺[j]*x̄⁺_a[:,:,j] + df.x⁻[j]*x̄⁻_a[:,:,j] 
    end
    ȳ_a = ȳ_a/luΦ̃ #convert to spline coefficients
    FO.ȳ_a = reshape(ȳ_a,:,n.sp)
    Eȳ′_a = ȳ_a*Φ̃ᵉ
    x̄_a = zeros(n.x,n.a,n.sp)
    for j in 1:n.sp
        #println("Computing f for j = ", j)
        @views f[j] = -inv(dF.x[j] + dF.yᵉ[j]*Eȳ′_a[:,:,j]*p)
        @views x̄_a[:,:,j] .= f[j]*dF.a[j]
    end
    FO.x̄_a = reshape(x̄_a,:,n.sp)/Φ̃
end


"""
    compute_Lemma3!(FO)

Computes the terms from Lemma 3, x_s = dx_t/dX_t+s
"""
function compute_Lemma3!(FO::FirstOrderApproximation)
    #compute_f_matrices!(FO) #maybe uncomment and then remove from compute_theta_derivative
    @unpack ZO,f,x,T = FO
    @unpack x̄,Φ̃ᵉ,Φ̃,p,n,dF,df,Δ⁺,Δ⁻ = ZO
    #N = length(ẑ)
    luΦ̃ = lu(Φ̃) #precompute inverse of basis matrix
    xtemp = zeros(n.x,n.Q,n.sp) #one nx x nQ matrix for each gridpoint
    ytemp = zeros(n.y,n.Q,n.sp) #one nx x nQ matrix for each gridpoint
    cFy′ = Vector{Matrix{Float64}}(undef,n.sp)
    
    for i in 1:n.sp
        cFy′[i] = f[i]*dF.yᵉ[i]
        xtemp[:,:,i] .= f[i]*dF.X[i]
    end
    x[1] = xtemp/luΦ̃

    for s in 2:T
        xtemp⁺ = xtemp*Δ⁺
        xtemp⁻ = xtemp*Δ⁻
        for i in 1:n.sp
            ytemp[:,:,i] .= df.x⁺[i]*xtemp⁺[:,:,i] .+ df.x⁻[i]*xtemp⁻[:,:,i]
        end
        Ey′ = ytemp/luΦ̃*Φ̃ᵉ
        for i in 1:n.sp
            @views xtemp[:,:,i] .= cFy′[i]*Ey′[:,:,i]
        end
        x[s] = xtemp/luΦ̃
    end
end


"""
    compute_Lemma4!(FO)

Computes the terms from Lemma 4, Operators L and terms a_s = M p x_s 
"""
function compute_Lemma4!(FO)
    @unpack ZO,x,T,x̄_a = FO
    @unpack Φ,Φₐ,Δ,p,pκ,ω̄,Λ,x̄,n,dlΓ = ZO
    #Λ is other the full joint distribution
    #Γ and dlΓ are vectors.
    #(x̄*Δ) computes the coefficients for x̄Δ= x̄⁻ - x̄⁺
    #compute Γκ̄
    
    #start with computing La 
    #ā_a = reshape((p*x̄)*Φₐ,n.a,:)  #ā_a is n.a x (n.a x n.Ω )(compress last two dimensions)
    ā_a =  reshape(p*reshape(x̄_a,n.x,n.a,:)*Φ,n.a,:)
    FO.La = La = kron(Λ,ones(n.a,n.a)) #initialize La operator 

    rowsLa = rowvals(La)
    for j in eachindex(ā_a[1,:]) #iterate over all columns
        for index in nzrange(La,j)
            i = rowsLa[index]
            ia = (i-1)%n.a+1
            @inbounds La.nzval[index] *= ā_a[ia,j]
        end
    end

    #Next the L operator
    κ̄_a = reshape((pκ*x̄_a)*Δ*Φ,:)
    dlΓκ̄_a = (dlΓ'.*reshape(κ̄_a,n.a,:))[:]
    FO.L = kron(Λ,ones(1,n.a)).*dlΓκ̄_a' #initialize L operator

    #Next compute a objects
    #Iz = length(ω̄)
    Ina = Matrix(I,n.a,n.a)
    ΛIna = kron(Λ,Ina)
    FO.Ma = ΛIna*kron(Φ'.*ω̄,Ina)
    FO.M =  Λ*(dlΓ.*ω̄.*Φ')
    #M is now (n.a x n.Ω) x (n.a x n.sp) matrix
    
    FO.a = a = zeros(n.sp*n.a,n.Q,T)
    FO.κ = κ = zeros(n.sp,n.Q,T)
    for s in 1:T
        #x[s] is n.x x n.Q x n.sp
        as = permutedims(p*x[s],[1,3,2]) #n.a x n.sp x n.Q
        @views a[:,:,s] .= reshape(as,:,n.Q)
        κs = permutedims(pκ*x[s]*Δ,[1,3,2]) #n.sp x n.Q
        @views κ[:,:,s] .= reshape(κs,:,n.Q)
    end

    #and now the I operators
    FO.I  = x̄*Φ #aggregate changes in density
    dlΓκ̄_a_mat = kron(sparse(I,n.Ω,n.Ω),ones(1,n.a)).*dlΓκ̄_a'
 
    FO.Ia = x̄_a*Φ .+ (x̄*Φ)*dlΓκ̄_a_mat   #aggregate changes in state 
end

"""
    compute_Corollary2!(FO)

Constructs J object from Corollary 1 
"""
function compute_Corollary2!(FO)
    @unpack ZO,T = FO
    @unpack Φₐ,x̄,ω̄ ,n,Φ,Δ = ZO
    Lat = sparse(FO.La')
    Lt = sparse(FO.L')
    Λt = sparse(ZO.Λ')
    Mt = sparse(FO.M')
    Mat = sparse(FO.Ma')
    #MΦ̃ = Φ.*ω̄'
    #Iz = n.z̄
    #compute expectations vector
    FO.IL = IL  = zeros(n.Ω*n.a,n.x,T-1)
    FO.IΛ = IΛ = zeros(n.Ω,n.x,T-1)
    IL[:,:,1] = (FO.Ia)'
    IΛ[:,:,1] = (FO.I)' 
    for t in 2:T-1
        @views IL[:,:,t] = Lat*IL[:,:,t-1] .+ Lt*IΛ[:,:,t-1]
        @views IΛ[:,:,t] = Λt*IΛ[:,:,t-1]
    end
    #transform in to  
    # Mt*IL is a (n.a x n.Ω)x n.x x T-1 matrix 
    FO.ILM = permutedims(Mat*IL,[2,3,1])
    FO.IΛM = permutedims(Mt*IΛ,[2,3,1])
    FO.IΛ = permutedims(IΛ,[2,3,1])
    #ILM is now n.x x T-1 x (n.a x n.Ω)
end 


function compute_Proposition1!(FO)
    #compute_Corollary2!(FO)
    @unpack ZO,x,T,J,a,κ,ILM,IΛM = FO
    @unpack Φ,Δ,p,pκ,ω̄ ,n,dlΓ,x̄= ZO

    #Iz = length(ω̄)
    #IA = n.x x T-1 x n.Q x T
    IA = ILM*a + IΛM*κ#reshape(reshape(ILM,n.ẑ,:)'*reshape(z,n.ẑ,:),n.x,T,n.Q,T)

    IntΦ̃ = Φ * ω̄ #operator to integrate splines over ergodic
    xvec = x̄*Φ
    IntΦ̃κ =(xvec.*ω̄'.*dlΓ')*Φ' #operator to integrate κ[s] over ergodic
    for s in 1:T
        κs = reshape(pκ*x[s]*Δ,n.Q,:)' #n.sp*2 x n.Q
        @views J[:,1,:,s] .= x[s]*IntΦ̃ .+ IntΦ̃κ*κs
    end

    #initialize l = 0
    for t in 2:T
        @views J[:,t,:,1] .= IA[:,t-1,:,1]
    end
    for s in 2:T
        for t in 2:T
            @views J[:,t,:,s] .= J[:,t-1,:,s-1] .+ IA[:,t-1,:,s]  
        end
    end
end

"""
    compute_BB!(FO::FirstOrderApproximation)

Computes the BB matrix
"""
function compute_BB!(FO::FirstOrderApproximation)
    @unpack ZO,T,J = FO
    @unpack dG,P,Q,n = ZO
    ITT = sparse(I,T,T)
    ITT_ = spdiagm(-1=>ones(T-1))
    ITTᵉ = spdiagm(1=>ones(T-1))
    #construct BB matrix
    FO.BB = kron(ITT,dG.x)*reshape(J,n.x*T,:)*kron(ITT,Q) .+ kron(ITT,dG.X) .+ kron(ITTᵉ,dG.Xᵉ) .+ kron(ITT_,dG.X_*P);
    FO.luBB = lu(FO.BB)
end



"""
    solve_Xt!(FO::FirstOrderApproximation)

Solves for the path Xt.
"""
function solve_Xt!(FO::FirstOrderApproximation)
    @unpack ZO,T,Θ_0,Δ_0,X_0,luBB,L = FO
    @unpack x̄,Φ,dG,n,Λ,ρ_Θ = ZO

    AA = zeros(n.X,T)
    for t in 1:T
        @views AA[:,t] .+= dG.Θ*ρ_Θ^(t-1)*Θ_0
    end
    IΛΔ_0 = [FO.IΛ*Δ_0 FO.IΛ[:,end,:]*Λ*Δ_0]
    AA .+= dG.x*IΛΔ_0
    

    AA[:,1] .+= dG.X_*X_0

    Xt = -(luBB\AA[:])
    FO.X̂t = reshape(Xt,n.X,T)
end


"""
    compute_x̂t_ω̂t!(FO::FirstOrderApproximation) 

Computes the x̂t and ω̂t objects.  Note assumes Na = 1. 
"""
function compute_x̂t_ω̂t!(FO::FirstOrderApproximation)
    @unpack ZO,T,x,κ,X̂t,L,M,La,Ma,Δ_0 = FO
    @unpack Q,Φ,ω̄,p,pκ,Δ,Δ⁺,Δ⁻,n,Λ,df,Φ̃ = ZO
    luΦ̃ = lu(Φ̃)
    #Fill objects
    #N = length(ZO.â)
    FO.x̂t = [zeros(n.x,n.sp) for t in 1:T]
    κ̂t = FO.κ̂t = [zeros(n.sp) for t in 1:T]
    QX̂t= Q*X̂t

    for s in 1:T
        x_s = permutedims(x[s],[1,3,2])
        κ_s = @view κ[:,:,s]
        for t in 1:T-(s-1)
            @views FO.x̂t[t] .+= x_s * QX̂t[:,t+s-1]
            @views FO.κ̂t[t] .+= κ_s * QX̂t[:,t+s-1]
        end
    end

    FO.ŷt = ŷt = [zeros(n.y,n.sp) for t in 1:T]
    ytemp = zeros(n.y,n.sp)
    for t in 1:T
        x̂t⁺ = FO.x̂t[t]*Δ⁺       
        x̂t⁻ = FO.x̂t[t]*Δ⁻
        for j in 1:n.sp
            @views ytemp[:,j] .= df.x⁺[j]*x̂t⁺[:,j] .+ df.x⁻[j]*x̂t⁻[:,j]
        end
        @views ŷt[t] .= ytemp/luΦ̃
    end

    #Next use ā_Z to construct Ω̂t
    ω̂t = FO.ω̂t = zeros(n.Ω,T)
    ω̂at = FO.ω̂at = zeros(n.Ω,T)
    ω̂t[:,1] = Δ_0
    for t in 2:T
        ât_t = (p*FO.x̂t[t-1])[:]
        κ̂t_t = κ̂t[t-1]
        @views ω̂at[:,t] = La*ω̂at[:,t-1] .+ Ma*ât_t 
        @views ω̂t[:,t] = Λ*ω̂t[:,t-1] .+ L*ω̂at[:,t-1]  .+ M*κ̂t_t
    end
end

function compute_p̂t!(FO::FirstOrderApproximation)
    @unpack ZO,T,x,κ,X̂t,L,M,La,Ma,Δ_0 = FO
    @unpack Q,Φ,ω̄,p,pκ,Δ,Δ⁺,Δ⁻,n,Λ,df,Φ̃,dΓ̃ = ZO
    luΦ̃ = lu(Φ̃)
    
    p̂t = FO.p̂t = zeros(n.sp,T)
    for t in 1:T
        κ̂t_t = κ̂t[t]*Φ̃
        p̂t[:,t] .= dΓ̃.*κ̂t_t
    end
end

"""
    compute_Ixt!(FO::FirstOrderApproximation)

Computes the path of integrated x_t values (Ixt) using x̂t and ω̂t.
This computes the aggregate changes in x over time using the distribution weights.

# Arguments
- `FO::FirstOrderApproximation`: The FirstOrderApproximation object containing x̂t and ω̂t

# Returns
- Updates Ixt with the computed path of integrated x values
"""
function compute_Ixt(FO::FirstOrderApproximation)
    @unpack ZO, x̂t, ω̂t,κ̂t,ω̂at = FO
    @unpack n,dlΓ,Φ,Φₐ,ω̄,x̄ = ZO

    IntΦ̃ = Φ * ω̄ #operator to integrate splines over ergodic
    xvec = x̄*Φ
    IntΦ̃κ =(xvec.*ω̄'.*dlΓ')*Φ' #operator to integrate κ[s] over ergodic
    
    # Initialize Ixt as a matrix of size n.x × T
    Ixt = zeros(n.x, FO.T)
    
    # For each time period, compute the integrated x value
    for t in 1:FO.T

        @views Ixt[:,t] = x̂t[t]*IntΦ̃ + IntΦ̃κ*κ̂t[t] + FO.I*ω̂t[:,t] + FO.Ia*ω̂at[:,t]
    end

    return Ixt
end


