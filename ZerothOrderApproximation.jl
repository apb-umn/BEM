#=============================================================================
    ZerothOrderApproximation.jl   (BEM: z-variable formulation WITH DISCRETE
    CHOICE — the port of the MPC code, New Approximation Code/
    ZerothOrderApproximationMPC.jl, to the occupational-choice model)

    The individual variables x are split into two groups
        x = (z, y),   z = (a′, forward-looking),   y = intra-temporal,
    ordered so that the z-variables are the FIRST n.z rows of xlab and the
    F equations are ordered with the n.z "z equations" (the intertemporal
    conditions that determine z) first, followed by the n.y intra-temporal
    equations that determine y given (a₋, θ, c, z, QX) with NO expectations.

    DISCRETE CHOICE (the BEM/BEMY structure).  Every (a, θ) node carries
    n.c = 2 occupations c ∈ {1,2}; the policy grid is the (a,θ) tensor grid
    replicated for each c, with c the SLOWEST index:
        full sparse node   j  = jr + (c-1)·n.spr,   jr ∈ 1:n.spr,
        n.sp = n.c·n.spr,  n.spr = n_ab·n.θ.
    Two user functions describe the choice (both read only the xelab rows,
    checked at build time):
      * f(x⁻, x⁺)  → yᵉ (length n.ye): the taste-shock-integrated variables
                     whose conditional expectations enter F (in BEM the
                     expected marginal utility and the log-sum-exp value);
                     x⁻ = x at the c=1 node, x⁺ = x at the c=2 node of the
                     SAME (a,θ).  F has signature
                        F(j, θ⃗, a⃗, c, x, QX, Eyᵉ′)
                     with Eyᵉ′ = E[f(x⁻′,x⁺′) | a′, θ] (length n.ye).
      * pf(x⁻, x⁺) → probability of choosing c = 1 at (a,θ).

    The DISTRIBUTION is tracked PRE-CHOICE on the reduced (a,θ) fine grid:
        μ_t(a,θ)   mass entering period t,    ω_t(a,θ,c) = p_{c,t}(a,θ) μ_t(a,θ),
    so that the steady-state transition on μ is a standard factored matrix
        Λ̃ = Σ_c Λc[c]·diag(p̄_c),
    with Λc[c] the user-supplied transition CONDITIONAL on occupation c (the
    per-θ lottery/interpolation blocks of the policy a′(·,θ,c) composed with
    πθ).  This halves the distribution size relative to the old (a,θ,c)
    representation and is exactly the timing of OCModelEGM.jl's dist!.

    Selectors (as in the MPC code):
      * zlab  ⊂ xlab : the z-variables, FIRST n.z entries of xlab; alab the
                       first n.a entries of zlab.
      * xelab ⊂ xlab : the variables f and pf READ (n.e of them).
      * yᵉlab        : names of the n.ye outputs of f.
      * Ilab  ⊂ xlab : the variables whose INTEGRALS G reads (checked).

    Operators built here (all value-to-value, PCHIP on tensor grids, stored
    as FactoredTransitionMatrix objects — see PCHIP.jl and the note
    ApproximationChanges.tex):
      Φ̃ₐ, Φ̃ᵉ, Φ̃ᵉₐ, Φ, Φₐ   on the FULL policy grid (n.c·n.θ blocks); the
                             expectation operators read the c = 1 copy of
                             the (duplicated) taste-integrated function,
                             Π_e = kron([1 0; 1 0], πθ);
      Φr                     sparse→fine value map on the REDUCED grid;
      Λ (= Λ̃), D[d], D2[d,e] on the reduced fine grid (n.θ blocks).

    Relative to the old BEM code (reference/ZerothOrderApproximation.jl):
      * x̄ holds policy VALUES at the sparse nodes (no spline coefficients);
      * only FIRST derivatives of F, G, f, pf are stored (the second order
        is built by AD on the fly, as in the MPC code);
      * the distribution lift uses the finite-difference D̄ operators of the
        MPC code instead of the Φₐ/La/Ma machinery, so first-order results
        agree with the old code only up to discretization.

    PRECISION / THREADING: identical conventions to the MPC code — the whole
    pipeline is parameterized on Tf ∈ {Float64, Float32} (Inputs{Tf}), and
    every hot per-node loop runs through `threaded_foreach` (PCHIP.jl);
    `julia -t 1` is the exact serial path.
=============================================================================#

using Parameters, SparseArrays, SuiteSparse, LinearAlgebra, ForwardDiff

include("PCHIP.jl")


"""
Nums stores sizes of various objects

a    number of endogenous idiosyncratic states (n.a)
x    number of individual policy functions
z    number of tracked (state + forward-looking) individual variables
y    number of intra-temporal individual variables (n.x - n.z)
e    number of individual variables read by f / pf (xelab)
ye   number of taste-integrated variables whose expectations enter F
I    number of individual variables whose integrals enter G
X    number of aggregate policy functions
Q    number of aggregate variables that appear in individual problem
A    number of aggregate predetermined variables
Ω    number of points in the REDUCED (a,θ) fine grid (where μ lives)
sp   number of points in the FULL sparse policy grid (= c·spr)
spr  number of points in the reduced (a,θ) sparse grid
θ    number of exogenous idiosyncratic states
c    number of discrete choices (2)
Θ    number of aggregate shocks
"""
@with_kw mutable struct Nums
    x::Int64   = 0
    z::Int64   = 0
    y::Int64   = 0
    e::Int64   = 0
    ye::Int64  = 0
    I::Int64   = 0
    X::Int64   = 0
    Q::Int64   = 0
    A::Int64   = 0
    Ω::Int64   = 0
    sp::Int64  = 0
    spr::Int64 = 0
    θ::Int64   = 0
    c::Int64   = 2
    Θ::Int64   = 0
    a::Int64   = 0
end


"""
Inputs is a struct that contains the user inputs, all at the base
floating-point type Tf (`Inputs()` defaults to Float64).  Assigning Float64
arrays to the fields of an `Inputs{Float32}` rounds them automatically.
"""
@with_kw mutable struct Inputs{Tf<:AbstractFloat}
    ## Policy VALUES at the FULL sparse nodes (n.x × n.sp, c slowest)
    x̄::Matrix{Tf}           = zeros(Tf,1,1)
    πθ::Matrix{Tf}          = ones(Tf,1,1)
    xlab::Vector{Symbol}    = Symbol[]
    alab::Vector{Symbol}    = Symbol[]      # endogenous states (prefix of zlab)
    zlab::Vector{Symbol}    = Symbol[]      # tracked variables (prefix of xlab)
    xelab::Vector{Symbol}   = Symbol[]      # variables read by f and pf
    yᵉlab::Vector{Symbol}   = Symbol[]      # outputs of f (expectations entering F)
    Ilab::Vector{Symbol}    = Symbol[]      # variables whose integrals G reads

    ## REDUCED (a,θ) grids — no choice column; the policy grid is the sparse
    ## grid replicated n.c times with c slowest (see the file header)
    aθ_sp::Matrix{Tf}       = zeros(Tf,1,1) # reduced sparse grid (policies)
    aθ_Ω::Matrix{Tf}        = zeros(Tf,1,1) # reduced fine grid (distribution)
    ℵ::Vector{Int}          = Int[]         # kink indices (dim-1 cutoffs)

    ## Steady-state distribution (pre-choice masses μ̄ and the probability
    ## p̄ of choice 1, both on the reduced fine grid) and the transition
    ## operators CONDITIONAL on each choice (reduced fine → reduced fine)
    μ̄::Vector{Tf}           = zeros(Tf,1)
    p̄::Vector{Tf}           = zeros(Tf,1)
    Λc::Vector{FactoredTransitionMatrix{Matrix{Tf},Tf}} = FactoredTransitionMatrix{Matrix{Tf},Tf}[]

    X̄::Vector{Tf}           = zeros(Tf,1)
    Xlab::Vector{Symbol}    = Symbol[]
    Alab::Vector{Symbol}    = Symbol[]
    Qlab::Vector{Symbol}    = Symbol[]

    ## Equilibrium definition.  F(j, θ⃗, a⃗, c, x, QX, Eyᵉ′) — j is the FULL
    ## sparse-node index, c the occupation, Eyᵉ′ the expected taste-
    ## integrated variables (length n.ye).  G keeps the full-Ix signature.
    ## f(x⁻,x⁺) → yᵉ and pf(x⁻,x⁺) → Prob(c = 1) describe the discrete choice.
    para::Any               = nothing
    F::Function             = (j,θ,a,c,x,QX,ye′)->zeros(1)
    G::Function             = (Ix,A_,X,Xᵉ,Θ)->zeros(1)
    f::Function             = (x⁻,x⁺)->zeros(1)
    pf::Function            = (x⁻,x⁺)->0.5

    ## Shock Process
    Θ̄::Vector{Tf}           = ones(Tf,1)
    ρ_Θ::Matrix{Tf}         = ones(Tf,1,1)
    Σ_Θ::Matrix{Tf}         = ones(Tf,1,1)
end

#  the historical un-parameterized call keeps working (Float64)
Inputs(; kwargs...) = Inputs{Float64}(; kwargs...)

"""
    _to_precision(Tf, inputs::Inputs)

Array-level precision conversion of an Inputs bundle: every numeric field is
rounded to Tf (grids included); labels pass through, and `para`/`F`/`G`/`f`/
`pf` are passed through UNCHANGED.
"""
function _to_precision(::Type{Tf}, inputs::Inputs{Ti}) where {Tf<:AbstractFloat,Ti}
    Tf === Ti && return inputs
    out = Inputs{Tf}()
    for name in fieldnames(Inputs)
        setproperty!(out, name, getfield(inputs, name))   # convert per field type
    end
    return out
end


"""
DerivativesF stores FIRST derivatives of F at each sparse node (eltype Tf):
    x[j]   n.x × n.x    (w.r.t. current x = (z,y))
    X[j]   n.x × n.Q    (w.r.t. the aggregate prices QX)
    xe[j]  n.x × n.ye   (w.r.t. the expectations Eyᵉ′)
    a[j]   n.x × n.a    (w.r.t. the lagged endogenous states)
No second derivatives are stored — second-order terms are AD'd on the fly.
"""
@with_kw mutable struct DerivativesF{Tf<:AbstractFloat}
    nsp::Int64 = 1
    x::Vector{Matrix{Tf}}      = Vector{Matrix{Tf}}(undef,nsp)
    X::Vector{Matrix{Tf}}      = Vector{Matrix{Tf}}(undef,nsp)
    xe::Vector{Matrix{Tf}}     = Vector{Matrix{Tf}}(undef,nsp)
    a::Vector{Matrix{Tf}}      = Vector{Matrix{Tf}}(undef,nsp)
end

"""
DerivativesDC stores the first derivatives of the discrete-choice maps at
each REDUCED sparse node jr (only the xelab columns — the others are zero):
    fm[:,:,jr]  n.ye × n.e   ∂f/∂x⁻[ie]        fp[:,:,jr]  ∂f/∂x⁺[ie]
    pm[:,jr]    n.e          ∂pf/∂x⁻[ie]       pp[:,jr]    ∂pf/∂x⁺[ie]
"""
@with_kw mutable struct DerivativesDC{Tf<:AbstractFloat}
    fm::Array{Tf,3} = zeros(Tf,0,0,0)
    fp::Array{Tf,3} = zeros(Tf,0,0,0)
    pm::Matrix{Tf}  = zeros(Tf,0,0)
    pp::Matrix{Tf}  = zeros(Tf,0,0)
end

"""
DerivativesG stores first derivatives of G (second-order G terms are AD'd)
"""
@with_kw mutable struct DerivativesG{Tf<:AbstractFloat}
    x::Matrix{Tf}      = zeros(Tf,1,1)
    X::Matrix{Tf}      = zeros(Tf,1,1)
    X_::Matrix{Tf}     = zeros(Tf,1,1)
    Xᵉ::Matrix{Tf}     = zeros(Tf,1,1)
    Θ::Matrix{Tf}      = zeros(Tf,1,1)
end


"""
The Zeroth order class (z-variable + discrete-choice variant).

Parameterized on the base floating-point type Tf: grids, operators and
derivatives all live in Tf.
"""
@with_kw mutable struct ZerothOrderApproximation{Tf<:AbstractFloat}
    n::Nums = Nums()

    # REDUCED grids (F receives the θ/a values of the reduced sparse grid)
    aθ_sp::Matrix{Tf} = zeros(Tf,1,1)
    aθ_Ω::Matrix{Tf}  = zeros(Tf,1,1)
    agrids_sp::Vector{Vector{Tf}} = Vector{Tf}[]  # per-dim sparse grids
    agrids_Ω::Vector{Vector{Tf}}  = Vector{Tf}[]  # per-dim fine grids

    # policy VALUES at the FULL sparse nodes (n.x × n.sp) and the
    # taste-integrated values ȳ = f(x̄⁻,x̄⁺) (n.ye × n.sp, duplicated over c)
    x̄::Matrix{Tf} = zeros(Tf,1,1)
    ȳ::Matrix{Tf} = zeros(Tf,1,1)

    # aggregates
    X̄::Vector{Tf} = zeros(Tf,1)

    # shock processes
    Θ̄::Vector{Tf}   = ones(Tf,1)
    ρ_Θ::Matrix{Tf} = Tf(0.8)*ones(Tf,1,1)
    Σ_Θ::Matrix{Tf} = Tf(0.014^2)*ones(Tf,1,1)

    # stationary distribution: pre-choice masses μ̄ and choice probability p̄
    # on the reduced fine grid; ω̄ = [p̄⊙μ̄; (1-p̄)⊙μ̄] on the full fine grid
    μ̄::Vector{Tf} = ones(Tf,1)
    p̄::Vector{Tf} = ones(Tf,1)
    ω̄::Vector{Tf} = ones(Tf,1)

    # basis / transition operators (all value-to-value maps)
    Φ̃ₐ::FactoredTransitionMatrix{UniformScaling{Bool},Tf}  = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=I)
    Φ̃ᵉ::FactoredTransitionMatrix{Matrix{Tf},Tf}            = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=ones(Tf,1,1))
    Φ̃ᵉₐ::FactoredTransitionMatrix{Matrix{Tf},Tf}           = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=ones(Tf,1,1))
    Φ::FactoredTransitionMatrix{UniformScaling{Bool},Tf}   = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=I)
    Φₐ::FactoredTransitionMatrix{UniformScaling{Bool},Tf}  = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=I)
    Φr::FactoredTransitionMatrix{UniformScaling{Bool},Tf}  = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=I)
    Λc::Vector{FactoredTransitionMatrix{Matrix{Tf},Tf}}    = FactoredTransitionMatrix{Matrix{Tf},Tf}[]
    Λ::FactoredTransitionMatrix{Matrix{Tf},Tf}             = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=ones(Tf,1,1))

    # finite-difference lift operators (per state, reduced fine grid)
    D::Vector{FactoredTransitionMatrix{UniformScaling{Bool},Tf}}  = FactoredTransitionMatrix{UniformScaling{Bool},Tf}[]
    D2::Matrix{FactoredTransitionMatrix{UniformScaling{Bool},Tf}} = Matrix{FactoredTransitionMatrix{UniformScaling{Bool},Tf}}(undef,0,0)

    # kinked policy rules (dim-1 cutoffs)
    ℵ::Vector{Int} = Int[]

    # selectors
    ie::Vector{Int} = Int[]           # positions of xelab within xlab
    iI::Vector{Int} = Int[]           # positions of Ilab within xlab
    p::Matrix{Tf} = zeros(Tf,1,1)     # n.a × n.x : ā′ = p x̄  (rows 1:n.a of x)
    P::Matrix{Tf} = zeros(Tf,1,1)     # X -> A_
    Q::Matrix{Tf} = zeros(Tf,1,1)     # X -> prices in HH problem

    # aggregation weights (integrals against the steady state)
    IntΦ::Vector{Tf} = zeros(Tf,1)    # n.sp : ∫ x̂ dω̄ = x̂·IntΦ
    Wp::Matrix{Tf}   = zeros(Tf,1,1)  # n.x × n.spr : ∫ (x̄₁-x̄₂) p̂ dμ̄ = Wp p̂
    Iμ::Matrix{Tf}   = zeros(Tf,1,1)  # n.x × n.Ω : ∫ x̄ dμ̂ = Iμ μ̂ (Σ_c p̄_c x̄_c at fine nodes)

    # F, G and discrete-choice derivatives
    dF::DerivativesF{Tf}   = DerivativesF{Tf}()
    dDC::DerivativesDC{Tf} = DerivativesDC{Tf}()
    dG::DerivativesG{Tf}   = DerivativesG{Tf}()
end


function construct_selector_matrix(n::Int64, indices::Vector)
    m = length(indices)
    return sparse(1:m, indices, 1, m, n)
end


#=============================================================================
    Tensor-grid helpers (identical to ZerothOrderApproximationMPC.jl)
=============================================================================#

"""
    tensor_grid(agrids, exo)

Build an aθ matrix (rows = grid points, columns = [a_1 … a_{n_a}, exo…]) in the
canonical ordering: dimension 1 fastest, …, exogenous state slowest.  `exo` is
an (nθ × n_exo) matrix, one row per exogenous state.
"""
function tensor_grid(agrids::Vector{<:AbstractVector}, exo::AbstractMatrix)
    lens = length.(agrids)
    n_ab = prod(lens)
    nθ   = size(exo,1)
    n_a  = length(agrids)
    Tv   = float(promote_type(eltype(eltype(agrids)), eltype(exo)))
    out  = zeros(Tv, n_ab*nθ, n_a + size(exo,2))
    for d in 1:n_a
        inner = prod(lens[1:d-1])
        outer = prod(lens[d+1:end])
        col = kron(ones(Tv,outer), kron(agrids[d], ones(Tv,inner)))
        out[:,d] = kron(ones(Tv,nθ), col)
    end
    for e in 1:size(exo,2)
        out[:,n_a+e] = kron(exo[:,e], ones(Tv,n_ab))
    end
    return out
end

"""
    endo_grids(aθ, n_a, nθ)

Extract the per-dimension endogenous grids from an aθ matrix and assert it has
the canonical tensor structure.
"""
function endo_grids(aθ::AbstractMatrix{Tv}, n_a::Int, nθ::Int) where {Tv<:AbstractFloat}
    N = size(aθ,1)
    @assert N % nθ == 0 "grid size not divisible by number of exogenous states"
    n_ab = N ÷ nθ
    grids = [unique(aθ[1:n_ab, d]) for d in 1:n_a]
    lens = length.(grids)
    @assert prod(lens) == n_ab "endogenous grid is not a tensor product (distinct nodes may have collided when rounding the grid to $(Tv))"
    for d in 1:n_a
        @assert issorted(grids[d]) "grid in dimension $d must be increasing"
        inner = prod(lens[1:d-1])
        outer = prod(lens[d+1:end])
        expected = kron(ones(Tv,outer), kron(grids[d], ones(Tv,inner)))
        @assert aθ[1:n_ab, d] == expected "grid ordering must be dim-1-fastest (see tensor_grid)"
    end
    for s in 2:nθ
        @assert view(aθ,(s-1)*n_ab+1:s*n_ab,1:n_a) == view(aθ,1:n_ab,1:n_a) "endogenous block must repeat identically across exogenous states"
    end
    return grids
end


"""
    rowwise_kron(A, B)

Row-wise Kronecker (Khatri–Rao) product: C[i,:] = kron(A[i,:], B[i,:]).
"""
function rowwise_kron(A::SparseMatrixCSC{Tv,Int64}, B::SparseMatrixCSC{Tv,Int64}) where {Tv}
    m = size(A,1)
    @assert size(B,1) == m
    q = size(B,2)
    At = sparse(A')
    Bt = sparse(B')
    rows = Int[]; cols = Int[]; vals = Tv[]
    sizehint!(rows, nnz(A)*6); sizehint!(cols, nnz(A)*6); sizehint!(vals, nnz(A)*6)
    for i in 1:m
        for ja in nzrange(At, i)
            colA = At.rowval[ja]; vA = At.nzval[ja]
            for jb in nzrange(Bt, i)
                colB = Bt.rowval[jb]; vB = Bt.nzval[jb]
                push!(rows, i); push!(cols, (colA-1)*q + colB); push!(vals, vA*vB)
            end
        end
    end
    return sparse(rows, cols, vals, m, size(A,2)*q)
end

"""
    tensor_eval_rows(agrids, pts)

Sparse matrix mapping values on the tensor grid `agrids` (dim 1 fastest) to
interpolated values at the scattered points `pts` (npts × n_a).
"""
function tensor_eval_rows(agrids::Vector{<:AbstractVector}, pts::AbstractMatrix)
    P = _eval_matrix(agrids[1], view(pts,:,1))
    for d in 2:length(agrids)
        P = rowwise_kron(_eval_matrix(agrids[d], view(pts,:,d)), P)
    end
    return P
end

"""
    tensor_deriv_rows(agrids, pts, dim)

Same as `tensor_eval_rows` but differentiates the interpolant in dimension
`dim` (evaluation in the other dimensions).
"""
function tensor_deriv_rows(agrids::Vector{<:AbstractVector}, pts::AbstractMatrix, dim::Int)
    mk(d) = d == dim ? _deriv_matrix(agrids[d], view(pts,:,d)) :
                       _eval_matrix(agrids[d], view(pts,:,d))
    P = mk(1)
    for d in 2:length(agrids)
        P = rowwise_kron(mk(d), P)
    end
    return P
end

"""
    tensor_kron(mats)

Full tensor-product operator on the flattened grid (dim 1 fastest).
"""
function tensor_kron(mats::Vector{SparseMatrixCSC{Tv,Int64}}) where {Tv}
    K = mats[1]
    for d in 2:length(mats)
        K = kron(mats[d], K)
    end
    return K
end


#=============================================================================
    Finite-difference lift operators D̄_d and D̄²_{de}
    (identical to ZerothOrderApproximationMPC.jl; built on the REDUCED fine
    grid, n.θ blocks)
=============================================================================#

function construct_D_1d(ā::AbstractVector{Tv}) where {Tv<:AbstractFloat}
    n_a = length(ā)
    h = ā[3:end] .- ā[1:end-2]

    d0 = zeros(Tv, n_a)
    d0[1]   = -1/(ā[2] - ā[1])
    d0[end] =  1/(ā[end] - ā[end-1])

    d1 = Vector{Tv}(undef, n_a-1)
    d1[1:end-1] .= -1 ./ h
    d1[end]      = -1/(ā[end] - ā[end-1])

    dm1 = Vector{Tv}(undef, n_a-1)
    dm1[1]      = 1/(ā[2] - ā[1])
    dm1[2:end] .= 1 ./ h

    return spdiagm(-1 => dm1, 0 => d0, 1 => d1)
end

function construct_D2_1d(ā::AbstractVector{Tv}) where {Tv<:AbstractFloat}
    n_a = length(ā)
    d0  = zeros(Tv, n_a)
    dp1 = zeros(Tv, n_a - 1)
    dp2 = zeros(Tv, n_a - 2)
    dm1 = zeros(Tv, n_a - 1)
    dm2 = zeros(Tv, n_a - 2)

    for j in 2:n_a-1
        hp = ā[j+1] - ā[j]
        hm = ā[j] - ā[j-1]
        hs = ā[j+1] - ā[j-1]
        dp1[j]   =  2 / (hs * hp)
        d0[j]    = -2 / (hp * hm)
        dm1[j-1] =  2 / (hs * hm)
    end

    # Left endpoint (j=1): one-sided using points 1,2,3
    h1 = ā[2] - ā[1]
    h2 = ā[3] - ā[2]
    dp2[1] = 2/(h2 * (h1 + h2))
    dp1[1] = -2/(h1 * h2)
    d0[1] = -dp2[1] - dp1[1]
    # Right endpoint (j=n_a): one-sided using points n_a-2, n_a-1, n_a
    h1 = ā[n_a] - ā[n_a-1]
    h2 = ā[n_a-1] - ā[n_a-2]
    dm2[end] = 2/(h2 * (h1 + h2))
    dm1[end] = -2/(h1 * h2)
    d0[end] = -dm2[end] - dm1[end]

    return sparse(spdiagm(-2 => dm2, -1 => dm1, 0 => d0, 1 => dp1, 2 => dp2)')
end

function construct_Ds(agrids_Ω::Vector{Vector{Tv}}, n::Nums) where {Tv<:AbstractFloat}
    lens = length.(agrids_Ω)
    D1   = [construct_D_1d(g) for g in agrids_Ω]
    eyes = [sparse(one(Tv)*I, l, l) for l in lens]
    D = Vector{FactoredTransitionMatrix{UniformScaling{Bool},Tv}}(undef, n.a)
    for dim in 1:n.a
        blk = tensor_kron([dim == d ? D1[d] : eyes[d] for d in 1:n.a])
        D[dim] = FactoredTransitionMatrix(Φs=[blk for _ in 1:n.θ], Π=I)
    end
    return D
end

function construct_D2s(agrids_Ω::Vector{Vector{Tv}}, n::Nums) where {Tv<:AbstractFloat}
    lens = length.(agrids_Ω)
    D1   = [construct_D_1d(g) for g in agrids_Ω]
    D2_1 = [construct_D2_1d(g) for g in agrids_Ω]
    eyes = [sparse(one(Tv)*I, l, l) for l in lens]
    D2 = Matrix{FactoredTransitionMatrix{UniformScaling{Bool},Tv}}(undef, n.a, n.a)
    for d in 1:n.a, e in 1:n.a
        mats = Vector{SparseMatrixCSC{Tv,Int64}}(undef, n.a)
        for c in 1:n.a
            if c == d == e
                mats[c] = D2_1[c]
            elseif c == d || c == e
                mats[c] = D1[c]
            else
                mats[c] = eyes[c]
            end
        end
        D2[d,e] = FactoredTransitionMatrix(Φs=[tensor_kron(mats) for _ in 1:n.θ], Π=I)
    end
    return D2
end


#=============================================================================
    PCHIP value-to-value operators with the discrete-choice block structure
=============================================================================#

"""
    interleave_rows(Ps)

Row-interleave the length-n.a vector of (m × n) matrices: row d + (i-1)·n.a
of the result is row i of Ps[d].
"""
function interleave_rows(Ps::Vector{SparseMatrixCSC{Tv,Int64}}) where {Tv}
    na = length(Ps)
    m  = size(Ps[1],1)
    σ  = vec(permutedims(reshape(1:na*m, m, na)))   # σ[d + (i-1)na] = i + (d-1)m
    return reduce(vcat, Ps)[σ, :]
end

"""
    construct_Φ̃s_pchip_dc(agrids_sp, agrids_Ω, ā′, πθ, n)

The PCHIP operators on the FULL policy grid (n.c·n.θ blocks, c slowest).
`ā′` (n.a × n.sp) are the savings chosen at every full node.  The
expectation operators read the c = 1 copy of the taste-integrated function
(which is duplicated over c), so their mixing matrix is
    Π_e = kron([1 0; 1 0], πθ)      (Π_e[(c,s),(c′,s′)] = πθ[s,s′]·1{c′=1}).
Also returns the reduced sparse→fine value map Φr (n.θ blocks).
"""
function construct_Φ̃s_pchip_dc(agrids_sp::Vector{Vector{Tv}},
                                agrids_Ω::Vector{Vector{Tv}},
                                ā′::AbstractMatrix{Tv},
                                πθ::Matrix{Tv}, n::Nums) where {Tv<:AbstractFloat}
    n_a = n.a
    nθ  = n.θ
    nc  = n.c
    S   = nθ*nc
    lens_sp = length.(agrids_sp)
    n_ab = prod(lens_sp)
    @assert n_ab*nθ == n.spr && n.spr*nc == n.sp
    tol = 1e-14
    eyes_sp = [sparse(one(Tv)*I, l, l) for l in lens_sp]

    # ---- Φ̃ₐ: derivatives at the sparse nodes, states interleaved ----
    Φ̃ₐ_blk = interleave_rows([tensor_kron([dim == d ?
                    droptol!(_deriv_matrix(agrids_sp[d], agrids_sp[d]), tol) :
                    eyes_sp[d] for d in 1:n_a]) for dim in 1:n_a])
    Φ̃ₐ_blk = sparse(Φ̃ₐ_blk')                     # RHS-multiplication orientation
    Φ̃ₐ = FactoredTransitionMatrix(Φs=[Φ̃ₐ_blk for _ in 1:S], Π=I)

    # ---- Φ̃ᵉ / Φ̃ᵉₐ: expectation operators at the chosen states ā′ ----
    Φ̃ᵉ_blocks  = Vector{SparseMatrixCSC{Tv,Int64}}(undef, S)
    Φ̃ᵉₐ_blocks = Vector{SparseMatrixCSC{Tv,Int64}}(undef, S)
    pts = Matrix{Tv}(undef, n_ab, n_a)
    for b in 1:S
        for d in 1:n_a
            pts[:,d] .= clamp.(view(ā′, d, (b-1)*n_ab+1:b*n_ab),
                               agrids_sp[d][1], agrids_sp[d][end])
        end
        Φ̃ᵉ_blocks[b] = droptol!(tensor_eval_rows(agrids_sp, pts), tol)
        Φ̃ᵉₐ_blocks[b] = interleave_rows([droptol!(tensor_deriv_rows(agrids_sp, pts, d), tol)
                                          for d in 1:n_a])
    end
    Πe = kron([1 0; 1 0], πθ)
    @assert nc == 2 "the discrete-choice operators are written for n.c = 2"
    Φ̃ᵉ  = transpose(FactoredTransitionMatrixTransposed(Φs=Φ̃ᵉ_blocks, Π=Matrix{Tv}(Πe)))
    Φ̃ᵉₐ = transpose(FactoredTransitionMatrixTransposed(Φs=Φ̃ᵉₐ_blocks, Π=Matrix{Tv}(Πe)))

    # ---- Φ: sparse -> fine values (full grid) and Φr (reduced grid) ----
    Φ_blk = tensor_kron([droptol!(_eval_matrix(agrids_sp[d], agrids_Ω[d]), tol) for d in 1:n_a])
    Φ_blk = sparse(Φ_blk')
    Φ  = FactoredTransitionMatrix(Φs=[copy(Φ_blk) for _ in 1:S], Π=I)
    Φr = FactoredTransitionMatrix(Φs=[copy(Φ_blk) for _ in 1:nθ], Π=I)

    # ---- Φₐ: sparse -> fine derivatives, states interleaved ----
    Φₐ_blk = interleave_rows([tensor_kron([dim == d ?
                    droptol!(_deriv_matrix(agrids_sp[d], agrids_Ω[d]), tol) :
                    droptol!(_eval_matrix(agrids_sp[d], agrids_Ω[d]), tol) for d in 1:n_a])
                              for dim in 1:n_a])
    Φₐ_blk = sparse(Φₐ_blk')
    Φₐ = FactoredTransitionMatrix(Φs=[copy(Φₐ_blk) for _ in 1:S], Π=I)

    return Φ̃ₐ, Φ̃ᵉ, Φ̃ᵉₐ, Φ, Φₐ, Φr
end


"""
    construct_Λ̃(Λc, p̄, n)

Steady-state transition of the PRE-CHOICE distribution μ on the reduced
fine grid: Λ̃ = Σ_c Λc[c]·diag(p̄_c) with p̄_1 = p̄, p̄_2 = 1 − p̄ (per-θ block:
Λc[1].Φs[s]·diag(p̄_s) + Λc[2].Φs[s]·diag(1−p̄_s), same mixing Π).
"""
function construct_Λ̃(Λc::Vector{FactoredTransitionMatrix{Matrix{Tv},Tv}}, p̄::Vector{Tv}, n::Nums) where {Tv}
    @assert length(Λc) == n.c == 2 "Λc must hold one conditional transition per choice (n.c = 2)"
    nθ = n.θ
    n_abΩ = n.Ω ÷ nθ
    for c in 1:n.c
        @assert length(Λc[c].Φs) == nθ "Λc[$c] must have n.θ blocks"
        @assert all(B -> size(B) == (n_abΩ, n_abΩ), Λc[c].Φs) "Λc[$c] blocks must be (n.Ω/n.θ)² on the reduced fine grid"
        @assert size(Λc[c].Π) == (nθ, nθ)
    end
    @assert Λc[1].Π == Λc[2].Π "the conditional transitions must share the exogenous mixing matrix"
    p̄r = reshape(p̄, n_abΩ, nθ)
    Φs = [Λc[1].Φs[s]*spdiagm(p̄r[:,s]) .+ Λc[2].Φs[s]*spdiagm(one(Tv) .- p̄r[:,s]) for s in 1:nθ]
    return FactoredTransitionMatrix(Φs=Φs, Π=Matrix{Tv}(Λc[1].Π))
end


"""
    ZerothOrderApproximation(inputs::Inputs{Tf}; Tf=…)

Build the zeroth-order object at the inputs' base floating-point type — all
operators are constructed FROM the Tf grids by the eltype-generic builders.
"""
function ZerothOrderApproximation(inputs::Inputs{Ti}; Tf::Type{<:AbstractFloat}=Ti) where {Ti}
    @assert Tf === Float64 || Tf === Float32 "only Float64 and Float32 are supported"
    Tf !== Ti && (inputs = _to_precision(Tf, inputs))
    @unpack x̄, aθ_sp, aθ_Ω, ℵ, μ̄, p̄, Λc, πθ, Θ̄, X̄ = inputs
    @unpack xlab, alab, zlab, xelab, yᵉlab, Ilab, Xlab, Alab, Qlab, ρ_Θ, Σ_Θ = inputs
    @unpack f, pf = inputs
    ZO = ZerothOrderApproximation{Tf}()

    nc = 2
    n = Nums(θ = size(πθ,1), c = nc, Ω = size(aθ_Ω,1), spr = size(aθ_sp,1), sp = nc*size(aθ_sp,1),
             x = length(xlab), z = length(zlab), y = length(xlab) - length(zlab),
             e = length(xelab), ye = length(yᵉlab), I = length(Ilab),
             X = length(X̄), Q = length(Qlab),
             A = length(Alab), Θ = length(Θ̄), a = length(alab))
    @assert size(x̄) == (n.x, n.sp) "x̄ must be n.x × n.sp (values at the FULL sparse nodes, c slowest)"
    @assert n.a >= 1
    @assert n.z >= n.a "zlab must at least contain the endogenous states"
    @assert length(μ̄) == n.Ω && length(p̄) == n.Ω "μ̄ and p̄ live on the reduced fine grid"
    @assert all(0 .<= p̄ .<= 1) "p̄ must be a probability"
    @assert isempty(ℵ) || all(j -> 1 <= j < n.sp, ℵ) "ℵ must hold interior sparse-node indices (1 ≤ j < n.sp)"

    # ---- ordering requirements of the z-formulation ----
    @assert xlab[1:n.z] == zlab "zlab must be the FIRST n.z entries of xlab (z-variables first)"
    @assert zlab[1:n.a] == alab "alab must be the FIRST n.a entries of zlab (states first within z)"
    ie = Int[findfirst(isequal(l), xlab) for l in xelab]
    iI = Int[findfirst(isequal(l), xlab) for l in Ilab]
    @assert !any(isnothing, ie) "every xelab entry must appear in xlab"
    @assert !any(isnothing, iI) "every Ilab entry must appear in xlab"

    iA = Int[findfirst(isequal(l), Xlab) for l in Alab]
    iQ = Int[findfirst(isequal(l), Xlab) for l in Qlab]

    agrids_sp = endo_grids(aθ_sp, n.a, n.θ)
    agrids_Ω  = endo_grids(aθ_Ω, n.a, n.θ)

    # next-period endogenous states chosen at each FULL node: rows 1:n.a of x̄
    ā′ = x̄[1:n.a, :]

    Φ̃ₐ, Φ̃ᵉ, Φ̃ᵉₐ, Φ, Φₐ, Φr = construct_Φ̃s_pchip_dc(agrids_sp, agrids_Ω, ā′, πθ, n)

    # taste-integrated steady-state values ȳ = f(x̄⁻, x̄⁺), duplicated over c
    ȳ = zeros(Tf, n.ye, n.sp)
    for jr in 1:n.spr
        yv = f(x̄[:, jr], x̄[:, jr + n.spr])
        @assert length(yv) == n.ye "f must return a vector of length n.ye = length(yᵉlab)"
        ȳ[:, jr] .= yv
        ȳ[:, jr + n.spr] .= yv
    end

    ZO.n = n
    ZO.aθ_sp = aθ_sp
    ZO.aθ_Ω = aθ_Ω
    ZO.agrids_sp = agrids_sp
    ZO.agrids_Ω = agrids_Ω
    ZO.x̄ = x̄
    ZO.ȳ = ȳ
    ZO.X̄ = X̄
    ZO.Θ̄ = Θ̄
    ZO.μ̄ = μ̄
    ZO.p̄ = p̄
    ZO.ω̄ = [p̄ .* μ̄; (one(Tf) .- p̄) .* μ̄]

    ZO.Φ̃ₐ = Φ̃ₐ
    ZO.Φ̃ᵉ = Φ̃ᵉ
    ZO.Φ̃ᵉₐ = Φ̃ᵉₐ
    ZO.Φ = Φ
    ZO.Φₐ = Φₐ
    ZO.Φr = Φr
    ZO.Λc = Λc
    ZO.Λ = construct_Λ̃(Λc, p̄, n)
    ZO.D = construct_Ds(agrids_Ω, n)
    ZO.D2 = construct_D2s(agrids_Ω, n)

    ZO.ie = ie
    ZO.iI = iI
    p = zeros(Tf, n.a, n.x)
    for d in 1:n.a
        p[d,d] = one(Tf)
    end
    ZO.p = p
    ZO.P = construct_selector_matrix(n.X, iA)
    ZO.Q = construct_selector_matrix(n.X, iQ)

    ZO.ℵ = ℵ
    ZO.ρ_Θ = ρ_Θ
    ZO.Σ_Θ = Σ_Θ

    # ---- aggregation weights ----
    #  ∫ x̂ dω̄ = Σ_j x̂[:,j]·IntΦ[j]         (full-grid quadrature weights)
    ZO.IntΦ = Φ * ZO.ω̄
    #  fine-grid policy values, x̄Φ[:, m, c]
    x̄Φ = reshape(x̄ * Φ, n.x, n.Ω, nc)
    #  ∫ x̄ dμ̂ = Iμ μ̂,  Iμ[:,m] = Σ_c p̄_c(m) x̄Φ[:,m,c]
    ZO.Iμ = x̄Φ[:,:,1] .* p̄' .+ x̄Φ[:,:,2] .* (one(Tf) .- p̄)'
    #  ∫ (x̄₁−x̄₂) p̂ dμ̄ = Wp p̂,  Wp = ((x̄Φ₁ − x̄Φ₂) ⊙ μ̄') Φrᵀ  (n.x × n.spr)
    ZO.Wp = ((x̄Φ[:,:,1] .- x̄Φ[:,:,2]) .* μ̄') * transpose(Φr)

    return ZO
end


"""
    stationarity_residual(ZO)

‖Λ̃ μ̄ − μ̄‖∞ — a diagnostic that the user-supplied conditional transitions
and choice probabilities reproduce the stationary distribution.
"""
stationarity_residual(ZO::ZerothOrderApproximation) = norm(ZO.Λ * ZO.μ̄ .- ZO.μ̄, Inf)


"""
    computeDerivativesF!(ZO, inputs)

FIRST derivatives of F at the steady state via ForwardDiff.  F has signature
F(j, θ⃗, a⃗, c, x, QX, Eyᵉ′) with j the FULL sparse-grid node index, c the
occupation, a⃗ (length n.a) the lagged endogenous states, θ⃗ the exogenous
column(s) of the reduced grid, and Eyᵉ′ the expected taste-integrated
variables (length n.ye).  Verifies that the y-block equations (rows
n.z+1:n.x) do not depend on the expectations — the defining property of the
z/y split.  Also computes the discrete-choice derivatives
(`computeDerivativesDC!`).  Returns the F residuals (n.x × n.sp).
"""
function computeDerivativesF!(ZO::ZerothOrderApproximation{Tf}, inputs::Inputs) where {Tf}
    @unpack n, aθ_sp, X̄, x̄, ȳ, Φ̃ᵉ, Q = ZO
    @unpack F = inputs
    ZO.dF = dF = DerivativesF{Tf}(nsp=n.sp)

    # Expected taste-integrated values at next-period states, E[ȳ′ | a′, θ]
    Eȳ′ = ȳ * Φ̃ᵉ
    argX̄ = Q*X̄

    Fres = zeros(Tf, n.x, n.sp)
    nt = n_tasks(n.sp)
    Fye_chunk = zeros(nt)
    threaded_foreach(n.sp, ntasks=nt) do ch, jr_range
        Fye_c = 0.0
        for j in jr_range
            c  = (j-1) ÷ n.spr + 1
            jr = j - (c-1)*n.spr
            a_ = aθ_sp[jr, 1:n.a]            # lagged endogenous states (Vector)
            θ  = aθ_sp[jr, n.a+1:end]        # exogenous state column(s) (Vector)
            argx̄   = x̄[:,j]
            argEȳ′ = Eȳ′[:,j]

            Fres[:,j] = F(j, θ, a_, c, argx̄, argX̄, argEȳ′)

            dF.a[j]  = ForwardDiff.jacobian(a->F(j,θ,a,c,argx̄,argX̄,argEȳ′), a_)
            dF.x[j]  = ForwardDiff.jacobian(x->F(j,θ,a_,c,x,argX̄,argEȳ′), argx̄)
            dF.xe[j] = ForwardDiff.jacobian(ye′->F(j,θ,a_,c,argx̄,argX̄,ye′), argEȳ′)
            dF.X[j]  = ForwardDiff.jacobian(X->F(j,θ,a_,c,argx̄,X,argEȳ′), argX̄)

            Fye = maximum(abs, @view dF.xe[j][n.z+1:end, :])
            Fye > Fye_c && (Fye_c = Fye)
        end
        Fye_chunk[ch] = Fye_c
    end
    Fye_max = maximum(Fye_chunk)
    @assert Fye_max < 1e-10 "y-block equations (rows n.z+1:n.x of F) depend on the expectations Eyᵉ′ (max |∂Fy/∂Eyᵉ′| = $Fye_max).  Any variable with a forward-looking equation must be listed in zlab and its equation ordered in the first n.z rows."

    computeDerivativesDC!(ZO, inputs)
    return Fres
end


"""
    computeDerivativesDC!(ZO, inputs)

First derivatives of the discrete-choice maps f(x⁻,x⁺) and pf(x⁻,x⁺) at
every reduced sparse node, restricted to the xelab columns.  Verifies that
neither map reads any other component of x.
"""
function computeDerivativesDC!(ZO::ZerothOrderApproximation{Tf}, inputs::Inputs) where {Tf}
    @unpack n, x̄, ie = ZO
    @unpack f, pf = inputs
    dDC = DerivativesDC{Tf}(fm = zeros(Tf, n.ye, n.e, n.spr), fp = zeros(Tf, n.ye, n.e, n.spr),
                            pm = zeros(Tf, n.e, n.spr), pp = zeros(Tf, n.e, n.spr))
    notE = setdiff(1:n.x, ie)
    nt = n_tasks(n.spr)
    off_chunk = zeros(nt)
    threaded_foreach(n.spr, ntasks=nt) do ch, jr_range
        off_c = 0.0
        for jr in jr_range
            xm = x̄[:, jr]
            xp = x̄[:, jr + n.spr]
            Jm = ForwardDiff.jacobian(x->f(x, xp), xm)
            Jp = ForwardDiff.jacobian(x->f(xm, x), xp)
            gm = ForwardDiff.gradient(x->pf(x, xp), xm)
            gp = ForwardDiff.gradient(x->pf(xm, x), xp)
            @assert size(Jm) == (n.ye, n.x)
            dDC.fm[:,:,jr] = Jm[:, ie]
            dDC.fp[:,:,jr] = Jp[:, ie]
            dDC.pm[:,jr]   = gm[ie]
            dDC.pp[:,jr]   = gp[ie]
            if !isempty(notE)
                off = max(maximum(abs, @view Jm[:, notE]), maximum(abs, @view Jp[:, notE]),
                          maximum(abs, @view gm[notE]), maximum(abs, @view gp[notE]))
                off > off_c && (off_c = off)
            end
        end
        off_chunk[ch] = off_c
    end
    offE = maximum(off_chunk)
    @assert offE < 1e-10 "f or pf depend on x-components outside xelab (max |∂/∂x| over non-xelab columns = $offE).  Add the offending rows to xelab."
    ZO.dDC = dDC
    return dDC
end


"""
    computeDerivativesG!(ZO, inputs)

First derivatives of G at the steady state.  Verifies that G's gradient in
the integrals Ix is zero outside the Ilab columns.  NOTE: Ilab must list
every x-row G actually READS, not merely those with nonzero gradients at the
steady state.
"""
function computeDerivativesG!(ZO::ZerothOrderApproximation{Tf}, inputs::Inputs) where {Tf}
    @unpack n, X̄, x̄, Φ, ω̄, Q, P, iI = ZO
    @unpack Θ̄, G = inputs

    dG = DerivativesG{Tf}()
    argΘ̄ = Θ̄

    X̄_ = P*X̄
    Ix̄ = x̄*(Φ*ω̄)

    Gres = G(Ix̄,X̄_,X̄,X̄,argΘ̄)
    dG.x  = ForwardDiff.jacobian(x->G(x,X̄_,X̄,X̄,argΘ̄), Ix̄)
    dG.X_ = ForwardDiff.jacobian(X_->G(Ix̄,X_,X̄,X̄,argΘ̄), X̄_)
    dG.X  = ForwardDiff.jacobian(X->G(Ix̄,X̄_,X,X̄,argΘ̄), X̄)
    dG.Xᵉ = ForwardDiff.jacobian(Xᵉ->G(Ix̄,X̄_,X̄,Xᵉ,argΘ̄), X̄)
    dG.Θ  = ForwardDiff.jacobian(Θ->G(Ix̄,X̄_,X̄,X̄,Θ), argΘ̄)

    notI = setdiff(1:n.x, iI)
    if !isempty(notI)
        offI = maximum(abs, @view dG.x[:, notI])
        @assert offI < 1e-10 "G depends on integrals outside Ilab (max |∂G/∂Ix| over non-Ilab columns = $offI).  Add the offending rows to Ilab."
    end

    ZO.dG = dG
    return Gres
end


#=============================================================================
    Discrete-choice linear maps used by the first (and second) order
=============================================================================#

"""
    taste_mul!(ŷ, x̂e, ZO)

Linearized taste-shock integration: for every reduced node jr and both
occupations c,
    ŷ[:, m, jr, c] = fm[:,:,jr]·x̂e[:, m, jr, 1] + fp[:,:,jr]·x̂e[:, m, jr, 2].
`x̂e` has layout (n.e, m…, n.sp) and `ŷ` (n.ye, m…, n.sp); the output is
duplicated over c (the expectation operators read the c = 1 copy).
"""
function taste_mul!(ŷ::AbstractArray, x̂e::AbstractArray, ZO::ZerothOrderApproximation)
    return taste_mul!(ŷ, x̂e, ZO.dDC.fm, ZO.dDC.fp, ZO.n)
end

#  raw form (explicit Jacobian arrays, any precision — used by the Float64
#  σσ kernel solves of the second order)
function taste_mul!(ŷ::AbstractArray, x̂e::AbstractArray, fm::AbstractArray{Tv,3},
                    fp::AbstractArray{Tv,3}, n::Nums) where {Tv}
    m = length(x̂e) ÷ (n.e * n.sp)
    @assert length(x̂e) == n.e*m*n.sp && length(ŷ) == n.ye*m*n.sp
    xr = reshape(x̂e, n.e, m, n.spr, n.c)
    yr = reshape(ŷ, n.ye, m, n.spr, n.c)
    threaded_foreach(n.spr) do _, jrs
        @inbounds for jr in jrs
            for q in 1:m, v in 1:n.ye
                s = zero(eltype(ŷ))
                for e in 1:n.e
                    s += fm[v,e,jr]*xr[e,q,jr,1] + fp[v,e,jr]*xr[e,q,jr,2]
                end
                yr[v,q,jr,1] = s
                yr[v,q,jr,2] = s
            end
        end
    end
    return ŷ
end

"""
    choice_prob!(p̂, x̂e, ZO)

Linearized choice probability at every reduced node:
    p̂[jr, m] = pm[:,jr]·x̂e[:, m, jr, 1] + pp[:,jr]·x̂e[:, m, jr, 2]
(`x̂e` layout (n.e, m…, n.sp); `p̂` layout (n.spr, m…), node index FIRST).
"""
function choice_prob!(p̂::AbstractArray, x̂e::AbstractArray, ZO::ZerothOrderApproximation)
    return choice_prob!(p̂, x̂e, ZO.dDC.pm, ZO.dDC.pp, ZO.n)
end

function choice_prob!(p̂::AbstractArray, x̂e::AbstractArray, pm::AbstractMatrix, pp::AbstractMatrix, n::Nums)
    m = length(x̂e) ÷ (n.e * n.sp)
    @assert length(x̂e) == n.e*m*n.sp && length(p̂) == n.spr*m
    xr = reshape(x̂e, n.e, m, n.spr, n.c)
    pr = reshape(p̂, n.spr, m)
    threaded_foreach(n.spr) do _, jrs
        @inbounds for jr in jrs
            for q in 1:m
                s = zero(eltype(p̂))
                for e in 1:n.e
                    s += pm[e,jr]*xr[e,q,jr,1] + pp[e,jr]*xr[e,q,jr,2]
                end
                pr[jr,q] = s
            end
        end
    end
    return p̂
end

"""
    expect_f!(Eŷ, x̂e, ybuf, ZO)

E[ŷ′] for a variation x̂e of the xelab rows: taste-integrate (`taste_mul!`
into the scratch `ybuf`, same size as `Eŷ`) and apply Φ̃ᵉ.  This is the
discrete-choice replacement of `x * Φ̃ᵉ`.
"""
function expect_f!(Eŷ::AbstractArray, x̂e::AbstractArray, ybuf::AbstractArray, ZO::ZerothOrderApproximation)
    taste_mul!(ybuf, x̂e, ZO)
    mul!(Eŷ, ybuf, ZO.Φ̃ᵉ)
    return Eŷ
end
