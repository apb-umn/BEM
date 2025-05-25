using Parameters,BasisMatrices,SparseArrays,SuiteSparse,TensorOperations,ForwardDiff


"""
Nums stores sizes of various objects

nx  number of individual policy function
nX  number of aggregate policy functions
nQ  number of aggregate variables that appear in individual problem
nA  number of aggregate predetermined variables 
nz̄  number of points on the for storing the histogram
nẑ  number of splines
nθ  number of points on shock grid
  
"""
@with_kw mutable struct Nums
    x::Int64   = 0
    y::Int64   = 0
    X::Int64   = 0    
    Q::Int64   = 0
    A::Int64   = 0
    Ω::Int64   = 0
    sp::Int64   = 0
    θ::Int64   = 0 
    Θ::Int64   = 0
    a::Int64   = 0

    #Portfolio
    R::Int64   = 0 
end


 """
Imputs is a stuct that contains the user inputs 
"""
@with_kw mutable struct Inputs

    ## Policy Functions
    #zeroth order policy functions
    xf::Vector{Function}    = [(a,θ,c)->0][:]
    dxf::Vector{Function}   = [(a,θ,c)->[0]][:]
    πθ::Matrix{Float64}     = zeros(1,1)
    xlab::Vector{Symbol}    = [:k,:λ,:v]    # notations from the paper
    alab::Vector{Symbol}    = [:k]          # notations from the paper
    κlab::Vector{Symbol}    = [:v]          # notations from the paper
    yᵉlab::Vector{Symbol}   = [:v]          # notations from the paper
    
    #basis functions
    aknots::Vector{Vector{Float64}}  = [zeros(1)] #knot points
    ka::Int                  = 2        #order of splines

    #gridpoints
    aθc_sp::Matrix{Float64}  = zeros(1,1) #course grid for splines a_sp
    aθc_Ω::Matrix{Float64}  = zeros(1,1) #fine grid for distribution a_Ω z̄
    #kinks
    ℵ::Vector{Int}          = Int[] #\aleph objects

    ## Steady State Aggregates
    #Distribution
    ω̄ ::Vector{Float64}     = zeros(1)
    dlΓ::Vector{Float64}     = zeros(1)
    Λ::SparseMatrixCSC{Float64,Int64}   = spzeros(1,1)

    #Aggregate variables
    X̄::Vector{Float64}      = zeros(1)
    # total vector of aggregate variables
    Xlab::Vector{Symbol}    = [:R,:W,:T,:ℐ,:C,:Y,:V,:K,:q]
#    Xlab::Vector{}    = [:K,:R,:W]

    # vector of past aggregate variables relevant for HH problem / G function / in matrix P
    Alab::Vector{Symbol}     = [:K,:q]
#    Alab::Vector{Symbol}    = [:K]

    # vector of present aggregate variables relevant for HH problem / F function / in matrix Q
    Qlab::Vector{Symbol}    = [:R,:W,:T]
#    Qlab::Vector{Symbol}    = [:R,:W]

    ##Stochastic Equilibrium
    #Equilibrium definition
    F::Function             = (para,θ,a,c,x,QX,x′)->zeros(1)
    G::Function             =  (para,Ix,A_,X,Xᵉ,Θ)->zeros(1)
    f::Function             = (para,x⁻,x⁺)->zeros(1)

    #Taste Shock details
    Γf::Function             = κ->0. #Gumbel Shock CDF
    dΓf::Function            = κ->0. #Gumbel Shock PDF

    #Shock Processes
    Θ̄::Vector{Float64}      =  ones(1)
    ρ_Θ::Matrix{Float64}     = 0.8*ones(1,1)
    Σ_Θ::Matrix{Float64}     = 0.014^2*ones(1,1)
 end


"""
DerivativesF stores derivatives of F 
"""


@with_kw mutable struct DerivativesF
    x::Vector{Matrix{Float64}}      =  [zeros(1,1)]
    X::Vector{Matrix{Float64}}      =  [zeros(1,1)]
    yᵉ::Vector{Matrix{Float64}}     =  [zeros(1,1)]
    a::Vector{Matrix{Float64}}      =  [zeros(1,1)]
    #aa::Vector{Array{Float64,3}}     =  [zeros(1,1,1)]
    #ax::Vector{Array{Float64,3}}     =  [zeros(1,1,1)]
    #ax′::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #xx::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #xX::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #xx′::Vector{Array{Float64,3}}   =  [zeros(1,1,1)]
    #Xx::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #XX::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #Xx′::Vector{Array{Float64,3}}   =  [zeros(1,1,1)]
    #x′x::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #x′X::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #x′x′::Vector{Array{Float64,3}}  =  [zeros(1,1,1)]
end


@with_kw mutable struct Derivativesf
    x⁺::Vector{Matrix{Float64}}      =  [zeros(1,1)]
    x⁻::Vector{Matrix{Float64}}      =  [zeros(1,1)]
    #aa::Vector{Array{Float64,3}}     =  [zeros(1,1,1)]
    #ax::Vector{Array{Float64,3}}     =  [zeros(1,1,1)]
    #ax′::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #xx::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #xX::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #xx′::Vector{Array{Float64,3}}   =  [zeros(1,1,1)]
    #Xx::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #XX::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #Xx′::Vector{Array{Float64,3}}   =  [zeros(1,1,1)]
    #x′x::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #x′X::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    #x′x′::Vector{Array{Float64,3}}  =  [zeros(1,1,1)]
end

"""
DerivativesG stores derivatives of G
"""

@with_kw mutable struct DerivativesG
    x::Matrix{Float64}      = zeros(1,1)
    X::Matrix{Float64}      = zeros(1,1) 
    X_::Matrix{Float64}     = zeros(1,1)
    Xᵉ::Matrix{Float64}      = zeros(1,1)
    Θ::Matrix{Float64}      = zeros(1,1)
    xx::Array{Float64,3}    = zeros(1,1,1)
    xX_::Array{Float64,3}   = zeros(1,1,1)
    xX::Array{Float64,3}    = zeros(1,1,1)
    xXᵉ::Array{Float64,3}    = zeros(1,1,1)
    xΘ::Array{Float64,3}    = zeros(1,1,1)
    X_X_::Array{Float64,3}  = zeros(1,1,1)
    X_X::Array{Float64,3}   = zeros(1,1,1)
    X_Xᵉ::Array{Float64,3}    = zeros(1,1,1)
    X_Θ::Array{Float64,3}   = zeros(1,1,1)
    XX::Array{Float64,3}    = zeros(1,1,1)
    XXᵉ::Array{Float64,3}    = zeros(1,1,1)
    XΘ::Array{Float64,3}    = zeros(1,1,1)
    XᵉXᵉ::Array{Float64,3}    = zeros(1,1,1)
    XᵉΘ::Array{Float64,3}    = zeros(1,1,1)
    ΘΘ::Array{Float64,3}    = zeros(1,1,1)
end



"""
The Zeroth order class that contains the objects that we need from the zeroth order 
"""


@with_kw mutable struct ZerothOrderApproximation
    # Nums
    n::Nums=Nums()
    # grids
    aθc_sp::Matrix{Float64}      = zeros(1,1) #course grid for splines 
    aθc_Ω::Matrix{Float64}      = zeros(1,1) #fine grid for distribution
    
    #policy functions
    x̄::Matrix{Float64} =  zeros(1,1) # coefficients for policy rules

    #aggregates
    X̄::Vector{Float64} = zeros(1) #steady state aggregates

    #Shock Processes
    Θ̄::Vector{Float64}      =  ones(1)
    ρ_Θ::Matrix{Float64} = 0.8*ones(1,1)
    Σ_Θ::Matrix{Float64} = 0.014^2*ones(1,1)
    μ_Θσσ::Vector{Float64} = zeros(1)
    ρΥ::Float64 = 0.75 #Persistance of volatility shock
    σΥ::Float64 = 1.  #Size of volatility shock
    Σ_ΘΥ::Matrix{Float64} = Σ_Θ#time varying risk premium

    #masses for the stationary distribution
    ω̄::Vector{Float64} = ones(1) #fraction of mass below jump
    dlΓ::Vector{Float64}     = zeros(1)

    #basis and transition matricies 
    Φ̃::SparseMatrixCSC{Float64,Int64}   = spzeros(n.sp,n.sp)
    Φ̃ₐ::SparseMatrixCSC{Float64,Int64}  = spzeros(n.sp,n.sp)
    Φ̃ᵉₐ::SparseMatrixCSC{Float64,Int64} = spzeros(n.sp,n.sp)
    Φ̃ᵉ::SparseMatrixCSC{Float64,Int64}  = spzeros(n.sp,n.sp)
    Φ::SparseMatrixCSC{Float64,Int64}  = spzeros(n.sp,n.sp)
    Φₐ::SparseMatrixCSC{Float64,Int64} = spzeros(n.sp,n.sp)
    Λ::SparseMatrixCSC{Float64,Int64}   = spzeros(n.Ω,n.Ω)
    Δ::SparseMatrixCSC{Float64,Int64}   = spzeros(n.sp,n.sp) #
    Δ⁺::SparseMatrixCSC{Float64,Int64}  = spzeros(n.sp,n.sp) #
    Δ⁻::SparseMatrixCSC{Float64,Int64}  = spzeros(n.sp,n.sp) #

    #kinked policy rules
    ℵ::Vector{Int}          = Int[] #ℵ objects
    
    #Objects for first order approximation
    p::Matrix{Float64} = zeros(1)'  #projection matrix
    pκ::Matrix{Float64} = zeros(1)'  #projection matrix
    P::Matrix{Float64} = zeros(1,1) #projection matrix X->A_ 
    Q::Matrix{Float64} = zeros(1,1) #selector matrix for prices relevant for HH problem
    Q′::Matrix{Float64} = zeros(1,1) #selector matrix for future aggregate variables
    
    # F and G 
    df::Derivativesf = Derivativesf()
    dF::DerivativesF = DerivativesF() 
    dG::DerivativesG = DerivativesG()

end


function create_array_with_one(n::Int, position::Int)
    arr = zeros(n)  # Create an array of zeros of length n
    arr[position] = 1  # Set the specified position to 1
    return arr
end


function construct_selector_matrix(n::Int64, indices::Vector)
    m = length(indices)
    sel_matrix = sparse(1:m, indices, 1, m, n)
    return sel_matrix
end




function construct_abasis(zgrid::Vector)::Basis{1, Tuple{SplineParams{Vector{Float64}}}}
    abasis = Basis(SplineParams(zgrid,0,2))
    return abasis
end
    


function construct_x̄s(abasis::Basis{1, Tuple{SplineParams{Vector{Float64}}}},xf::Matrix{Function},n::Nums)
    #@unpack iz = inputs
    x̄f = Matrix{Interpoland}(undef,n.x,n.θ,2) #allow for discrete choice
    x̄=zeros(n.x,n.sp,2)
    for c in 1:2
        for i in 1:n.x
            x̄f[i,:,c] .= [Interpoland(abasis,a->xf[i,c](a,s)) for s in 1:n.θ]
            x̄[i,:,c]  = hcat([x̄f[i,s,c].coefs' for s in 1:n.θ]...)
        end
    end
    return x̄
end

function construct_Φ̃s(abasis,aθc_sp::Matrix{Float64},aθc_Ω::Matrix{Float64},af::Vector{Function},vf::Function,Γf::Function,dΓf::Function,πθ::Matrix{Float64},n::Nums)
    aθ_sp = aθc_sp[:,1:end-1]
    aθ_Ω = aθc_Ω[:,1:end-1]
    
    a_sp = unique(aθ_sp[:,1:n.a],dims=1)
    θ    = unique(aθ_sp[:,n.a+1:end],dims=1)
    a_Ω  = unique(aθ_Ω[:,1:n.a],dims=1) 

    
    Ia = Matrix{Int}(I,n.a,n.a)
    Na = size(a_sp,1)
    #note n.sp = Na*n.θ
    I2 = Matrix{Int}(I,2,2)
    Φ̃ = kron(I2,kron(Matrix(I,n.θ,n.θ),BasisMatrix(abasis,Expanded(),a_sp).vals[1]))'
    Φ̃ₐvec = [BasisMatrix(abasis,Expanded(),a_sp,Ia[:,ia]).vals[1] for ia in 1:n.a] 
    Φ̃ₐtemp = spzeros(Na*n.a,Na)
    for ia in 1:n.a
        range = ia:n.a:Na*n.a
        Φ̃ₐtemp[range,:] = Φ̃ₐvec[ia]
    end
    Φ̃ₐ = kron(I2,kron(Matrix(I,n.θ,n.θ),Φ̃ₐtemp))'

    Φ̃ᵉcols = [spzeros(n.sp,n.sp) for i in 1:2*n.θ]
    #Φ̃ᵉₐcols = [spzeros(n.sp,n.sp) for i in 1:2*n.θ]
    for c in 1:2
        for s in 1:n.θ
            a′ = reshape([af[ia](θ[s,:],a_sp_i,c)[1] for a_sp_i in eachrow(a_sp) for ia in 1:n.a],n.a,:) #ideally an n.a x n.sp  matrix
            Φ̃ᵉtemp = BasisMatrix(abasis,Expanded(),a′').vals[1]
            #Φ̃ᵉₐvec = [BasisMatrix(abasis,Expanded(),a′',Ia[:,ia]).vals[1] for ia in 1:n.a]
            Φ̃ᵉrows = [spzeros(Na,Na) for i in 1:2*n.θ]
            #Φ̃ᵉₐrows = [spzeros(2*Na,Na) for i in 1:2*n.θ]
            for s′ in 1:n.θ
                #b′ = R̄*bgrid .+ ϵ[s]*W̄ .- cf[s](bgrid) #asset choice
                #Δv′ = [vf(θ[s′,:],a′_i,2)-vf(θ[s′,:],a′_i,1) for a′_i in eachcol(a′)] 
                #Δdv′ = hcat([dvf(θ[s′,:],a′_i,2)-dvf(θ[s′,:],a′_i,1) for a′_i in eachcol(a′)]...)'
                #p = Γf.(Δv′)
                #dΓvec = zeros(length(p))
                #pmask = 0.9999 .> p .> 0.0001 #only evaluate for p in (0,1)
                #dΓvec[pmask] .= dΓf.(Δv′[pmask])
                #dΓvec = dΓf.(Δv′)
                #dΓvec[isnan.(dΓvec)] .= 0
                #dp = dΓvec.*Δdv′
                if c == 1
                    Φ̃ᵉrows[s′] = πθ[s,s′]*Φ̃ᵉtemp
                elseif c == 2
                    Φ̃ᵉrows[n.θ+s′] = πθ[s,s′]*Φ̃ᵉtemp
                end
                
                #for ia in 1:n.a
                #    range_a= ia:n.a:Na*n.a
                #    Φ̃ᵉₐrows[s′][range_a,:] = πθ[s,s′]*(p.*Φ̃ᵉₐvec[ia] .+ dp[:,ia].*Φ̃ᵉtemp)
                #end
                #now c′ = 2
                #Φ̃ᵉrows[n.θ+s′] = πθ[s,s′]*(1 .- p).*Φ̃ᵉtemp
                #for ia in 1:n.a
                #    range_a= ia:n.a:Na*n.a
                #    Φ̃ᵉₐrows[n.θ+s′][range_a,:] = πθ[s,s′]*((1 .- p).*Φ̃ᵉₐvec[ia] .- dp[:,ia].*Φ̃ᵉtemp)
                #end
            end
            Φ̃ᵉcols[(c-1)*n.θ+s] = hcat(Φ̃ᵉrows...)
            #Φ̃ᵉₐcols[(c-1)*n.θ+s] = hcat(Φ̃ᵉₐrows...)
        end
    end
    Φ̃ᵉ = vcat(Φ̃ᵉcols...)'
    #Φ̃ᵉₐ = vcat(Φ̃ᵉₐcols...)'


    NaΩ = size(a_Ω,1)
    Φ = kron(I2,kron(Matrix(I,n.θ,n.θ),BasisMatrix(abasis,Expanded(),a_Ω).vals[1]))'
    Φₐvec = [BasisMatrix(abasis,Expanded(),a_Ω,Ia[:,ia]).vals[1] for ia in 1:n.a] 
    Φₐtemp = spzeros(NaΩ*n.a,Na)
    for ia in 1:n.a
        range = ia:n.a:NaΩ*n.a
        Φₐtemp[range,:] = Φₐvec[ia]
    end
    Φₐ = kron(I2,kron(Matrix(I,n.θ,n.θ),Φₐtemp))'


    #New Δ objects 
    Naθ = size(unique(aθ_sp,dims=1),1)
    Δ⁺= kron([0 1;0 1],sparse(I,Naθ,Naθ))'
    Δ⁻= kron([1 0;1 0],sparse(I,Naθ,Naθ))'
    Δ = Δ⁺-Δ⁻

    NaθΩ = size(unique(aθ_Ω,dims=1),1)
    dlΓvec = zeros(n.Ω)
    tol = 1e-9
    for i in 1:n.Ω
        avec,θvec,c = aθc_Ω[i,1:n.a],aθc_Ω[i,n.a+1:end-1],aθc_Ω[i,end]
        Δv′ = vf(θvec,avec,2)[1]-vf(θvec,avec,1)[1]

        p = Γf(Δv′)
        #dΓvec = zeros(length(p))
        #pmask = 0.9999 .> p .> 0.0001 #only evaluate for p in (0,1)
        #dΓvec[pmask] .= dΓf.(Δv′[pmask])
        dΓvec = dΓf(Δv′)
        if p > 1-tol || p < tol
            dlΓvec[i] = 0
        else
            if c == 1
                dlΓvec[i] = dΓvec/p
            else
                dlΓvec[i] = -dΓvec/(1-p)
            end
        end
    end
    return Φ̃,Φ̃ₐ,Φ̃ᵉ,Φ,Φₐ,Δ⁺,Δ⁻,Δ,dlΓvec
end 



function ZerothOrderApproximation(inputs::Inputs)
    @unpack xf,aknots,aθc_sp,aθc_Ω,ℵ,Γf,dΓf = inputs
    @unpack ω̄,Λ,dlΓ, πθ, X̄ = inputs    
    @unpack xlab, alab, κlab, Xlab, Alab, Qlab,yᵉlab = inputs
    @unpack ρ_Θ, Σ_Θ,Θ̄ = inputs
    ZO =    ZerothOrderApproximation()
    
    n=Nums(θ = size(πθ)[1], Ω = size(aθc_Ω,1), sp = size(aθc_sp,1), x=length(xf), X=length(Xlab), Q=length(Qlab),  A=length(Alab), Θ=length(Θ̄), a=length(alab),y=length(yᵉlab))

    ia = (xlab .== reshape(alab,1,:))' * (1:n.x)# [(xlab .== reshape(alab,1,:))' * (1:n.x) for i =1:length(alab)][1] #if we only have a single z variable
    iκ = ((xlab .== reshape(κlab,1,:))' * (1:n.x))[1] #[(xlab .== reshape(vlab,1,:))' * (1:n.x) for i =1:length(vlab)][1] #if we only have a single z variable
    iA = (Xlab .== reshape(Alab,1,:))' * (1:n.X) # [(Xlab .== Alab[i])' * (1:n.X) for i =1:length(Alab)]
    iQ = (Xlab .== reshape(Qlab,1,:))' * (1:n.X) #[(Xlab .== Qlab[i])' * (1:n.X) for i =1:length(Qlab)]

    abasis= Basis([SplineParams(aknots[i],0,inputs.ka) for i in 1:n.a]...)
    Φ̃,Φ̃ₐ,Φ̃ᵉ,Φ,Φₐ,Δ⁺,Δ⁻,Δ,dlΓvec = construct_Φ̃s(abasis,aθc_sp,aθc_Ω,xf[ia],xf[iκ],Γf,dΓf,πθ,n)

    #Now compute x̄
    x̂ = zeros(n.x,n.sp)
    for ix in 1:n.x
        for j in 1:n.sp
            a,θ,c = aθc_sp[j,1:n.a],aθc_sp[j,n.a+1:end-1],Int(aθc_sp[j,end])
            x̂[ix,j] = xf[ix](θ,a,c)[1]
        end
    end

    ZO.n=n
    ZO.aθc_sp = aθc_sp
    ZO.aθc_Ω = aθc_Ω
    ZO.x̄ = reshape(x̂,n.x,:)/Φ̃
    ZO.X̄ = X̄
    ZO.ω̄ = ω̄ 

    ZO.Φ̃ = Φ̃
    ZO.Φ̃ₐ = Φ̃ₐ
    ZO.Φ̃ᵉ = Φ̃ᵉ
    ZO.Φ = Φ
    ZO.Φₐ = Φₐ
    ZO.Δ = Δ
    ZO.Δ⁺ = Δ⁺
    ZO.Δ⁻ = Δ⁻
    ZO.Λ = Λ
    ZO.dlΓ = dlΓvec

    ZO.p = construct_selector_matrix(n.x,ia)
    ZO.pκ = create_array_with_one(n.x,iκ)'
    ZO.P = construct_selector_matrix(n.X,iA)
    ZO.Q = construct_selector_matrix(n.X,iQ)

    ZO.ℵ = ℵ

    ZO.ρ_Θ = ρ_Θ
    ZO.Σ_Θ = Σ_Θ
    ZO.Θ̄ = Θ̄
        
    return ZO
end



function computeDerivativesF!(ZO::ZerothOrderApproximation,inputs::Inputs)
    @unpack n,aθc_sp,Φ̃,X̄,x̄,Φ̃ᵉ,Q = ZO
    @unpack F,f = inputs
    dF = DerivativesF()
    dF.x  = [zeros(n.x,n.x) for _ in 1:n.sp]
    dF.yᵉ = [zeros(n.x,n.y) for _ in 1:n.sp]
    dF.X = [zeros(n.x,n.Q) for _ in 1:n.sp]
    dF.a  =[zeros(n.x,n.a) for _ in 1:n.sp]
    df = Derivativesf()
    df.x⁺ = [zeros(n.y,n.x) for _ in 1:n.sp]
    df.x⁻ = [zeros(n.y,n.x) for _ in 1:n.sp]
    #dF⁻.aa,dF⁺.aa=[zeros(n.x,n.a,n.a) for _ in 1:n.sp],[zeros(n.x,n.a,n.a) for _ in 1:n.sp]
    #dF⁻.ax,dF⁺.ax=[zeros(n.x,n.x,n.a) for _ in 1:n.sp],[zeros(n.x,n.x,n.a) for _ in 1:n.sp]
    #dF⁻.ax′,dF⁺.ax′=[zeros(n.x,n.x,n.a) for _ in 1:n.sp],[zeros(n.x,n.x,n.a) for _ in 1:n.sp]
    #dF⁻.xx,dF⁺.xx=[zeros(n.x,n.x,n.x) for _ in 1:n.sp],[zeros(n.x,n.x,n.x) for _ in 1:n.sp]
    #dF⁻.xX,dF⁺.xX=[zeros(n.x,n.x,n.Q) for _ in 1:n.sp],[zeros(n.x,n.x,n.Q) for _ in 1:n.sp]
    #dF⁻.xx′,dF⁺.xx′=[zeros(n.x,n.x,n.x) for _ in 1:n.sp],[zeros(n.x,n.x,n.x) for _ in 1:n.sp]
    #dF⁻.Xx,dF⁺.Xx=[zeros(n.x,n.Q,n.x) for _ in 1:n.sp],[zeros(n.x,n.Q,n.x) for _ in 1:n.sp]
    #dF⁻.XX,dF⁺.XX=[zeros(n.x,n.Q,n.Q) for _ in 1:n.sp],[zeros(n.x,n.Q,n.Q) for _ in 1:n.sp]
    #dF⁻.Xx′,dF⁺.Xx′=[zeros(n.x,n.Q,n.x) for _ in 1:n.sp],[zeros(n.x,n.Q,n.x) for _ in 1:n.sp]
    #dF⁻.x′x,dF⁺.x′x=[zeros(n.x,n.x,n.x) for _ in 1:n.sp],[zeros(n.x,n.x,n.x) for _ in 1:n.sp]
    #dF⁻.x′X,dF⁺.x′X=[zeros(n.x,n.x,n.Q) for _ in 1:n.sp],[zeros(n.x,n.x,n.Q) for _ in 1:n.sp]
    #dF⁻.x′x′,dF⁺.x′x′=[zeros(n.x,n.x,n.x) for _ in 1:n.sp],[zeros(n.x,n.x,n.x) for _ in 1:n.sp]

    x̄ = ZO.x̄*Φ̃
    x̄⁺ = x̄*ZO.Δ⁺ 
    x̄⁻ = x̄*ZO.Δ⁻
    ȳ = zeros(n.y,n.sp)
    for j in 1:n.sp
        ȳ[:,j] = inputs.f(x̄⁻[:,j],x̄⁺[:,j]) 
    end
    Eȳ′ = (ȳ/Φ̃)*Φ̃ᵉ
    argX̄= Q*X̄ #only interest rate and wages relevant
    for j in 1:n.sp
        a_,θ,c = aθc_sp[j,1:n.a],aθc_sp[j,n.a+1:end-1],aθc_sp[j,end]
        argx̄ = x̄[:,j]
        argx̄⁺ = x̄⁺[:,j]
        argx̄⁻ = x̄⁻[:,j]
        argEȳ′ = Eȳ′[:,j]
        F(θ,a_,c,argx̄,argX̄,argEȳ′)
        #region = inputs.region(θ,a_)
        # first order
        @views dF.a[j]      = ForwardDiff.jacobian(a->F(θ,a,c,argx̄,argX̄,argEȳ′),a_)
        @views dF.x[j]      = ForwardDiff.jacobian(x->F(θ,a_,c,x,argX̄,argEȳ′),argx̄)
        @views dF.yᵉ[j]     = ForwardDiff.jacobian(y′->F(θ,a_,c,argx̄,argX̄,y′),argEȳ′)
        @views dF.X[j]      = ForwardDiff.jacobian(X->F(θ,a_,c,argx̄,X,argEȳ′),argX̄)
        @views df.x⁻[j]    = ForwardDiff.jacobian(x->inputs.f(x,argx̄⁺),argx̄⁻)
        @views df.x⁺[j]    = ForwardDiff.jacobian(x->inputs.f(argx̄⁻,x),argx̄⁺)
        
        # second order
        #dF⁻.aa[j]     = reshape(ForwardDiff.jacobian(a2->ForwardDiff.jacobian(a1->F⁻(θ,a1,argx̄,argX̄,argEx̄′),a2),a_),n.x,n.a,n.a)
        #dF⁺.aa[j]     = reshape(ForwardDiff.jacobian(a2->ForwardDiff.jacobian(a1->F⁺(θ,a1,argx̄,argX̄,argEx̄′),a2),a_),n.x,n.a,n.a)
        #dF⁻.ax[j]     = reshape(ForwardDiff.jacobian(x->ForwardDiff.jacobian(a1->F⁻(θ,a1,x,argX̄,argEx̄′),a_),argx̄),n.x,n.a,n.x)
        #dF⁺.ax[j]     = reshape(ForwardDiff.jacobian(x->ForwardDiff.jacobian(a1->F⁺(θ,a1,x,argX̄,argEx̄′),a_),argx̄),n.x,n.a,n.x)
        #dF⁻.xa[j],dF⁺.xa[j] = permutedims(dF⁻.ax[j],[1,3,2]),permutedims(dF⁺.ax[j],[1,3,2])
        #dF⁻.ax′[j]    = reshape(ForwardDiff.jacobian(x′ -> ForwardDiff.jacobian(a1->F⁻(θ,a1,argx̄,argX̄,x′),a_),argEx̄′),n.x,n.a,n.x)
        #dF⁺.ax′[j]    = reshape(ForwardDiff.jacobian(x′ -> ForwardDiff.jacobian(a1->F⁺(θ,a1,argx̄,argX̄,x′),a_),argEx̄′),n.x,n.a,n.x)
        #dF⁻.x′a[j],dF⁺.x′a[j] = permutedims(dF⁻.ax′[j],[1,3,2]),permutedims(dF⁺.ax′[j],[1,3,2])
        #dF⁻.xx[j]     = reshape(ForwardDiff.jacobian(x1 -> ForwardDiff.jacobian(x2->F⁻(θ,a_,x2,argX̄,argEx̄′),x1),argx̄),n.x,n.x,n.x)
        #dF⁺.xx[j]     = reshape(ForwardDiff.jacobian(x1 -> ForwardDiff.jacobian(x2->F⁺(θ,a_,x2,argX̄,argEx̄′),x1),argx̄),n.x,n.x,n.x)
        #dF⁻.xX[j]     = reshape(ForwardDiff.jacobian(X -> ForwardDiff.jacobian(x->F⁻(θ,a_,x,X,argEx̄′),argx̄),argX̄),n.x,n.x,n.Q)
        #dF⁺.xX[j]     = reshape(ForwardDiff.jacobian(X -> ForwardDiff.jacobian(x->F⁺(θ,a_,x,X,argEx̄′),argx̄),argX̄),n.x,n.x,n.Q)
        #dF⁻.Xx[j],dF⁺.Xx[j] = permutedims(dF⁻.xX[j],[1,3,2]),permutedims(dF⁺.xX[j],[1,3,2])
        #dF⁻.XX[j]     = reshape(ForwardDiff.jacobian(X1 -> ForwardDiff.jacobian(X2->F⁻(θ,a_,argx̄,X2,argEx̄′),X1),argX̄),n.x,n.Q,n.Q)
        #dF⁺.XX[j]     = reshape(ForwardDiff.jacobian(X1 -> ForwardDiff.jacobian(X2->F⁺(θ,a_,argx̄,X2,argEx̄′),X1),argX̄),n.x,n.Q,n.Q)
        #dF⁻.Xx′[j]    = reshape(ForwardDiff.jacobian(X -> ForwardDiff.jacobian(x′->F⁻(θ,a_,argx̄,X,x′),argEx̄′),argX̄),n.x,n.Q,n.x)
        #dF⁺.Xx′[j]    = reshape(ForwardDiff.jacobian(X -> ForwardDiff.jacobian(x′->F⁺(θ,a_,argx̄,X,x′),argEx̄′),argX̄),n.x,n.Q,n.x)
        #dF⁻.x′X[j],dF⁺.x′X[j] = permutedims(dF⁻.Xx′[j],[1,3,2]),permutedims(dF⁺.Xx′[j],[1,3,2])
        #dF⁻.xx′[j]   = reshape(ForwardDiff.jacobian(x′ -> ForwardDiff.jacobian(x->F⁻(θ,a_,x,argX̄,x′),argx̄),argEx̄′),n.x,n.x,n.x)
        #dF⁺.xx′[j]   = reshape(ForwardDiff.jacobian(x′ -> ForwardDiff.jacobian(x->F⁺(θ,a_,x,argX̄,x′),argx̄),argEx̄′),n.x,n.x,n.x)
        #dF⁻.x′x[j],dF⁺.x′x[j] = permutedims(dF⁻.xx′[j],[1,3,2]),permutedims(dF⁺.xx′[j],[1,3,2])
        #dF⁻.x′x′[j]  = reshape(ForwardDiff.jacobian(x1′ -> ForwardDiff.jacobian(x2′->F(θ,a_,argx̄,argX̄,x2′),x1′),argEx̄′),n.x,n.x,n.x)
        #dF⁺.x′x′[j]  = reshape(ForwardDiff.jacobian(x1′ -> ForwardDiff.jacobian(x2′->F(θ,a_,argx̄,argX̄,x2′),x1′),argEx̄′),n.x,n.x,n.x)
    end
   
    ZO.dF=dF;
    ZO.df=df;
end


function computeDerivativesG!(ZO::ZerothOrderApproximation,inputs::Inputs)
    #construct G derivatives
    @unpack n, X̄, x̄, Φ,Q,P,Θ̄ = ZO
    @unpack ω̄, G = inputs
    dG = DerivativesG()
    argΘ̄=Θ̄[1]

    X̄_ = P*X̄
    Ix̄ = x̄*Φ*ω̄

    #first order
    dG.x = ForwardDiff.jacobian(x->G(x,X̄_,X̄,X̄,[argΘ̄]),Ix̄) 
    dG.X_ = ForwardDiff.jacobian(X_->G(Ix̄,X_,X̄,X̄,[argΘ̄]),X̄_) 
    dG.X = ForwardDiff.jacobian(X->G(Ix̄,X̄_,X,X̄,[argΘ̄]),X̄) 
    dG.Xᵉ= ForwardDiff.jacobian(Xᵉ->G(Ix̄,X̄_,X̄,Xᵉ,[argΘ̄]),X̄)
    dG.Θ = ForwardDiff.jacobian(Θ->G(Ix̄,X̄_,X̄,X̄,Θ),[argΘ̄])

    #second order
    #dG.xx   = reshape(ForwardDiff.jacobian(x2->ForwardDiff.jacobian(x1->G(x1,X̄_,X̄,X̄,[argΘ̄]),x2),Ix̄),n.X,n.x,n.x)
    #dG.xX_  = reshape(ForwardDiff.jacobian(X_->ForwardDiff.jacobian(x->G(x,X_,X̄,X̄,[argΘ̄]),Ix̄),X̄_),n.X,n.x,n.A)
    #dG.xX   = reshape(ForwardDiff.jacobian(X->ForwardDiff.jacobian(x->G(x,X̄_,X,X̄,[argΘ̄]),Ix̄),X̄),n.X,n.x,n.X)
    #dG.xXᵉ   = reshape(ForwardDiff.jacobian(Xᵉ->ForwardDiff.jacobian(x->G(x,X̄_,X̄,Xᵉ,[argΘ̄]),Ix̄),X̄),n.X,n.x,n.X)
    #dG.xΘ   = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(x->G(x,X̄_,X̄,X̄,Θ),Ix̄),[argΘ̄]),n.X,n.x,n.Θ)
    #dG.X_X_ = reshape(ForwardDiff.jacobian(X2_->ForwardDiff.jacobian(X1_->G(Ix̄,X1_,X̄,X̄,[argΘ̄]),X2_),X̄_),n.X,n.A,n.A)
    #dG.X_X  = reshape(ForwardDiff.jacobian(X->ForwardDiff.jacobian(X_->G(Ix̄,X_,X,X̄,Θ̄),X̄_),X̄),n.X,n.A,n.X)
    #dG.X_Xᵉ = reshape(ForwardDiff.jacobian(Xᵉ->ForwardDiff.jacobian(X_->G(Ix̄,X_,X̄,Xᵉ,Θ̄),X̄_),X̄),n.X,n.A,n.X)
    #dG.X_Θ  = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(X_->G(Ix̄,X_,X̄,X̄,Θ),X̄_),[argΘ̄]),n.X,n.A,n.Θ)
    #dG.XX   = reshape(ForwardDiff.jacobian(X2->ForwardDiff.jacobian(X1->G(Ix̄,X̄_,X1,X̄,[argΘ̄]),X2),X̄),n.X,n.X,n.X)
    #dG.XXᵉ  = reshape(ForwardDiff.jacobian(Xᵉ->ForwardDiff.jacobian(X->G(Ix̄,X̄_,X,Xᵉ,[argΘ̄]),X̄),X̄),n.X,n.X,n.X)
    #dG.XΘ   = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(X->G(Ix̄,X̄_,X,X̄,Θ),X̄),[argΘ̄]),n.X,n.X,n.Θ)
    #dG.XᵉXᵉ   = reshape(ForwardDiff.jacobian(Xᵉ2->ForwardDiff.jacobian(Xᵉ1->G(Ix̄,X̄_,X̄,Xᵉ1,[argΘ̄]),Xᵉ2),X̄),n.X,n.X,n.X)
    #dG.ΘΘ   = reshape(ForwardDiff.jacobian(Θ2->ForwardDiff.jacobian(Θ1->G(Ix̄,X̄_,X̄,X̄,Θ1),Θ2),[argΘ̄]),n.X,n.Θ,n.Θ)

    #fixed weird forward diff bug for this derivative when $Xᵉ$ is not used
    #dG.XᵉΘ   = permutedims(reshape(ForwardDiff.jacobian(Xᵉ->ForwardDiff.jacobian(Θ->G(Ix̄,X̄_,X̄,Xᵉ,Θ),[argΘ̄]),X̄),n.X,n.Θ,n.X),[1,3,2])

    ZO.dG=dG;
    
end


function CheckG(ZO::ZerothOrderApproximation,inputs::Inputs)
    #construct G derivatives
    @unpack n, X̄, x̄, Φ,Q,P,Θ̄ = ZO
    @unpack ω̄, G = inputs
    argΘ̄=Θ̄[1]
    X̄_ = P*X̄
    Ix̄ = x̄*Φ*ω̄

    ret=G(Ix̄,X̄_,X̄,X̄,[argΘ̄])
    return ret
end



function CheckF(ZO::ZerothOrderApproximation,inputs::Inputs)
    @unpack n,aθc_sp,Φ̃,X̄,x̄,Φ̃ᵉ,Q = ZO
    @unpack F,f = inputs
    
    x̄ = ZO.x̄*Φ̃
    x̄⁺ = x̄*ZO.Δ⁺ 
    x̄⁻ = x̄*ZO.Δ⁻
    ȳ = zeros(n.y,n.sp)
    res= zeros(n.sp,n.x)
    for j in 1:n.sp
        ȳ[:,j] = inputs.f(x̄⁻[:,j],x̄⁺[:,j]) 
    end
    Eȳ′ = (ȳ/Φ̃)*Φ̃ᵉ
    argX̄= Q*X̄ #only interest rate and wages relevant
    for j in 1:n.sp
        a_,θ,c = aθc_sp[j,1:n.a],aθc_sp[j,n.a+1:end-1],aθc_sp[j,end]
        argx̄ = x̄[:,j]
        argx̄⁺ = x̄⁺[:,j]
        argx̄⁻ = x̄⁻[:,j]
        argEȳ′ = Eȳ′[:,j]
        res[j,:]=F(θ,a_,c,argx̄,argX̄,argEȳ′)
    end
   return res
end
