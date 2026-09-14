#=============================================================================
    FirstOrderApproximation.jl   (BEM: z-variable formulation WITH DISCRETE
    CHOICE — the port of New Approximation Code/FirstOrderApproximationMPC.jl)

    Pairs with ZerothOrderApproximation.jl (z/y split, discrete-choice
    conventions, selectors).  Everything below is the MPC first order with
    three discrete-choice additions, marked (DC):

    ------------------------------------------------------------------------
    THEORY (all per FULL node j; blocks of dF with z-equations/z-variables
    first):

    Intratemporal block (rows n.z+1:n.x of F, no expectations):
        ŷ_t = y_z ẑ_t + y_X QX̂_t,     y_z = −Fyy⁻¹Fyz,  y_X = −Fyy⁻¹FyQ,
    so the full x̂ is reconstructed as
        x̂_t = Wz ẑ_t + WX QX̂_t,       Wz = [I; y_z],  WX = [0; y_X].

    Intertemporal block (rows 1:n.z), with Eȳ_a = ∂E[ȳ′]/∂a′ (from Φ̃ᵉₐ
    applied to the taste-integrated steady-state values ȳ):
        0 = (Fzz + Fzy y_z + Fze Eȳ_a p_z) ẑ_t + Fze E_f[x̂e_{t+1}]
            + (FzQ + Fzy y_X) QX̂_t
        ⇒  ẑ_t = fFze·E_f[x̂e_{t+1}] + fF̃zQ·QX̂_t,        f = −D⁻¹,
    where x̂e = Sez ẑ + SeX QX̂ (rows ie of Wz/WX) and (DC)
        E_f[x̂e] = ( fm·x̂e(·,c=1) + fp·x̂e(·,c=2) ) Φ̃ᵉ
    is the linearized taste integration followed by the conditional
    expectation (`expect_f!`).  Kernels z_s and the backward recursion for
    ẑ_t are exactly as in the MPC code; only the savings rows (FO.a), the
    (DC) choice-probability rows (FO.pk, p̂ = pm·x̂e(·,1) + pp·x̂e(·,2)) and
    the ω̄-integrated Ilab rows (FO.Iz) survive the kernel pass.

    Distribution (DC): the PRE-CHOICE masses μ̂_t on the reduced fine grid,
        μ̂_{t+1} = Λ̃ μ̂_t + Σ_{d,c} M[d,c]·â_{d,t}(·,c) + Mp·p̂_t,
        M[d,c] = D̄_d Λc[c] diag(ω̄_c) Φrᵀ,      Mp = (Λc[1] − Λc[2]) diag(μ̄) Φrᵀ,
    i.e. one finite-difference lift per (state, occupation) — the displaced
    mass of occupation c is pushed forward with THAT occupation's policy —
    plus the mass reassigned between occupations by the probability change.
    Aggregation needs only values (DC: Wp and Iμ from the ZO):
        ∫ x dω̂_t = x̂_t·IntΦ + Wp p̂_t + Iμ μ̂_t.

    J matrix (Proposition 1, n.I rows only): row 1 is FO.Iz (kernel integrals
    incl. the Wp·pk term and the direct ∫WX dω̄ term on the s=1 slab); the
    rest accumulates the streamed IA tensor
        IA[t,s] = Iμ Λ̃^{t-1} ( Σ_{d,c} M[d,c] a_{d,c,s} + Mp pk_s ),
    materialized (Z idea p.3) as ONE transposed stack over the flattened
    kernel index (d,j) ∪ (jr):
        ILMt[k,i,t] = [Iμ Λ̃^{t-1} M_k]_{i,·},  k ∈ 1:(n.a·n.sp + n.spr),
    so IA falls out of a single gemm against the stacked kernels.

    THREADING / PRECISION: as in the MPC code (threaded_foreach over node
    chunks, BLAS pinned to 1 thread except for the huge gemms/LUs, Tf-generic).
=============================================================================#

using Parameters, SparseArrays, SuiteSparse, LinearAlgebra
BLAS.set_num_threads(1)
include("ZerothOrderApproximation.jl")

"""
    _with_blas_threads(f)

Run `f()` with BLAS temporarily widened to the Julia session's thread count
(for the handful of huge dense gemms/LUs).  No-op in a single-threaded session.
"""
function _with_blas_threads(f::F) where {F}
    nt = Threads.nthreads()
    nt == 1 && return f()
    nblas = BLAS.get_num_threads()
    BLAS.set_num_threads(nt)
    try
        return f()
    finally
        BLAS.set_num_threads(nblas)
    end
end


# ============================================================
# Array contraction helpers (no Base piracy)
# ============================================================

"""
    mullast(A, B)

Contract the LAST dimension of `A` with the first dimension of `B`
(B a vector or matrix).  Result has A's leading dims then B's trailing dims.
"""
@inline function mullast(A::AbstractArray, B::AbstractVecOrMat)
    sA, sB = size(A), size(B)
    return reshape(reshape(A,:,sA[end])*reshape(B,sB[1],:), sA[1:end-1]..., sB[2:end]...)
end

"""
    mulfirst(A, B)

Contract the matrix `A` with the FIRST dimension of the array `B`.
"""
@inline function mulfirst(A::AbstractMatrix, B::AbstractArray)
    sB = size(B)
    return reshape(A*reshape(B,sB[1],:), size(A,1), sB[2:end]...)
end

"""
    contract2(A, u, v)

Contract the 3-tensor `A` with vectors `u` (dim 2) and `v` (dim 3).
"""
@inline function contract2(A::AbstractArray{<:Real,3}, u::AbstractVector, v::AbstractVector)
    sA = size(A)
    return reshape(reshape(A,:,sA[3])*v, :, sA[2])*u
end


# ============================================================
# In-place operation helpers (as in FirstOrderApproximationMPC.jl)
# ============================================================

"""
    @doinplace A .= B * C    →  array_mul!(A, B, C)
    @doinplace A .+= B * C   →  array_mul!(A, B, C, 1, 1)
"""
macro doinplace(ex)
    if !(ex isa Expr && ex.head in (:.=,:.+=)
          && ex.args[2] isa Expr && ex.args[2].head == :call)
        error("@doinplace expects `A .= B * C` or `A .+= B * C`")
    end
    A = ex.args[1]
    rhs = ex.args[2]
    f = rhs.args[1]
    if f == :*
        B, C = rhs.args[2], rhs.args[3]
        if ex.head == :.=
            return esc(:(array_mul!($A, $B, $C)))
        else
            return esc(:(array_mul!($A, $B, $C, one(eltype($A)), one(eltype($C)))))
        end
    else
        error("@doinplace expects `A .= B * C` or `A .+= B * C`")
    end
end

# B matrix applied to the FIRST dimension of C
@inline function array_mul!(A::AbstractArray, B::AbstractMatrix, C::AbstractArray)
    mul!(reshape(A,size(A,1),:), B, reshape(C,size(C,1),:))
end
@inline function array_mul!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    mul!(A, B, C)
end
@inline function array_mul!(A::AbstractVector, B::AbstractMatrix, C::AbstractVector)
    mul!(A, B, C)
end
# B array contracted on its LAST dimension with matrix/vector C
@inline function array_mul!(A::AbstractArray, B::AbstractArray, C::AbstractMatrix)
    mul!(reshape(A,:,size(A)[end]), reshape(B,:,size(B)[end]), C)
end
@inline function array_mul!(A::AbstractArray, B::AbstractArray, C::AbstractVector)
    mul!(reshape(A,:), reshape(B,:,size(B)[end]), C)
end
@inline function array_mul!(A::AbstractArray, B::AbstractArray, C::AbstractVector, α::Real, β::Real)
    mul!(reshape(A,:), reshape(B,:,size(B)[end]), C, α, β)
end
@inline function array_mul!(A::AbstractVector, B::AbstractMatrix, C::AbstractVector, α::Real, β::Real)
    mul!(A, B, C, α, β)
end
@inline function array_mul!(A::AbstractArray, B::AbstractMatrix, C::AbstractArray, α::Real, β::Real)
    mul!(reshape(A,size(A,1),:), B, reshape(C,size(C,1),:), α, β)
end
@inline function array_mul!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, α::Real, β::Real)
    mul!(A, B, C, α, β)
end
# FactoredTransitionMatrix variants
@inline function array_mul!(A::AbstractArray, B::AbstractArray, C::FactoredTransitionMatrix)
    mul!(A, B, C)
end
@inline function array_mul!(A::AbstractMatrix, B::FactoredTransitionMatrixTransposed, C::AbstractMatrix)
    mul!(A, B, C)
end
@inline function array_mul!(A::AbstractVector, B::FactoredTransitionMatrix, C::AbstractVector)
    mul!(A, B, C)
end
@inline function array_mul!(A::AbstractMatrix, B::FactoredTransitionMatrix, C::AbstractMatrix)
    mul!(A, B, C)
end
@inline function array_mul!(A::AbstractVector, B::FactoredTransitionMatrix, C::AbstractVector, α::Real, β::Real)
    mul!(A, B, C, α, β)
end
@inline function array_mul!(A::AbstractMatrix, B::FactoredTransitionMatrix, C::AbstractMatrix, α::Real, β::Real)
    mul!(A, B, C, α, β)
end


"""
FirstOrderApproximation (z-variable + discrete-choice variant)

Per-node small matrices (node index LAST so slices drop into matrix products):
  y_z:    n.y × n.z × n.sp    intratemporal ŷ = y_z ẑ + y_X QX̂
  y_X:    n.y × n.Q × n.sp
  invFyy: n.y × n.y × n.sp    −invFyy·H_y gives the second-order y forcing
  f:      n.z × n.z × n.sp    −D⁻¹ (z-block Schur complement)
  fFze:   n.z × n.ye × n.sp   f·Fze   (recursion kernel)
  fF̃zQ:  n.z × n.Q × n.sp    f·(FzQ + Fzy y_X)  ( = kernel z_0 )
  Wz/WX:  n.x × {n.z,n.Q} × n.sp   full-x reconstruction stacks
  Sez/SeX, IzW/IXW: the xelab / Ilab row-slices of Wz/WX (contiguous copies)
  Eȳ_a:   n.ye × n.a × n.sp   ∂E[ȳ′]/∂a′ at every node
  x̄_a:   n.x × n.a × n.sp    full policy Jacobian in the lagged states

Kernel-pass outputs (kernels themselves are NOT stored):
  a:      n.a × n.sp × n.Q × T   savings kernels
  pk:     n.spr × n.Q × T        choice-probability kernels (DC)
  Iz:     n.I × n.Q × T          integral rows of the kernels (J row 1)
  ILMt:   (n.a·n.sp + n.spr) × n.I × (T-1)   the streamed Iμ Λ̃^{t-1} [M | Mp]
  IA, J:  n.I × (T-1|T) × n.Q × T

Path outputs:  ẑt (n.z × n.sp × T), p̂t (n.spr × T), Ω̂t (n.Ω × T, the
pre-choice mass perturbation μ̂), X̂t (n.X × T); full x̂ is reconstructed on
demand via `reconstruct_rows!`.
"""
@with_kw mutable struct FirstOrderApproximation{Tf<:AbstractFloat}
    ZO::ZerothOrderApproximation{Tf}
    T::Int #Length of IRF

    #Derivative direction
    Ω_0::Vector{Tf} = zeros(Tf,1)   # initial μ̂_1 (reduced fine grid)
    X_0::Vector{Tf} = zeros(Tf,1)
    Θ_0::Vector{Tf} = zeros(Tf,1)

    #per-node first-order objects
    y_z::Array{Tf,3}    = zeros(Tf,0,0,0)
    y_X::Array{Tf,3}    = zeros(Tf,0,0,0)
    invFyy::Array{Tf,3} = zeros(Tf,0,0,0)
    f::Array{Tf,3}      = zeros(Tf,0,0,0)
    fFze::Array{Tf,3}   = zeros(Tf,0,0,0)
    fF̃zQ::Array{Tf,3}  = zeros(Tf,0,0,0)
    Wz::Array{Tf,3}     = zeros(Tf,0,0,0)
    WX::Array{Tf,3}     = zeros(Tf,0,0,0)
    Sez::Array{Tf,3}    = zeros(Tf,0,0,0)
    SeX::Array{Tf,3}    = zeros(Tf,0,0,0)
    IzW::Array{Tf,3}    = zeros(Tf,0,0,0)
    IXW::Array{Tf,3}    = zeros(Tf,0,0,0)
    Eȳ_a::Array{Tf,3}   = zeros(Tf,0,0,0)
    x̄_a::Array{Tf,3}   = zeros(Tf,0,0,0)

    #Kernel-pass outputs (Lemma 3 / Lemma 4)
    a::Array{Tf,4}      = zeros(Tf,0,0,0,0)
    pk::Array{Tf,3}     = zeros(Tf,0,0,0)
    Iz::Array{Tf,3}     = zeros(Tf,0,0,0)
    L::FactoredTransitionMatrix{Matrix{Tf},Tf} = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=ones(Tf,1,1))
    Lt::FactoredTransitionMatrixTransposed{Matrix{Tf},Tf} = FactoredTransitionMatrixTransposed(Φs=[spzeros(Tf,1,1)], Π=ones(Tf,1,1))
    M::Matrix{FactoredTransitionMatrix{Matrix{Tf},Tf}} = Matrix{FactoredTransitionMatrix{Matrix{Tf},Tf}}(undef,0,0)
    Mt::Matrix{FactoredTransitionMatrixTransposed{Matrix{Tf},Tf}} = Matrix{FactoredTransitionMatrixTransposed{Matrix{Tf},Tf}}(undef,0,0)
    Mp::FactoredTransitionMatrix{Matrix{Tf},Tf} = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=ones(Tf,1,1))
    Mpt::FactoredTransitionMatrixTransposed{Matrix{Tf},Tf} = FactoredTransitionMatrixTransposed(Φs=[spzeros(Tf,1,1)], Π=ones(Tf,1,1))

    #streamed IA pieces and the J matrix (Proposition 1)
    ILMt::Array{Tf,3} = zeros(Tf,1,1,1)
    IA::Array{Tf,4} = zeros(Tf,1,1,1,1)
    J::Array{Tf,4} = zeros(Tf,1,1,1,1)

    #Terms for Proposition 1 (luBB abstractly typed: dense LU in Tf, or the
    #sparse fallback promoted to Float64 — UMFPACK supports no other precision)
    BB::SparseMatrixCSC{Tf, Int64} = spzeros(Tf,1,1)
    luBB::Factorization = lu(ones(Tf,1,1))

    #Outputs
    Ω̂t::Matrix{Tf} = zeros(Tf,1,1)
    ẑt::Array{Tf,3} = zeros(Tf,1,1,1)
    p̂t::Matrix{Tf} = zeros(Tf,1,1)
    X̂t::Matrix{Tf} = zeros(Tf,1,1)

    Ω̂_Θt::Vector{Matrix{Tf}} = Vector{Matrix{Tf}}(undef,1)
    ẑ_Θt::Vector{Array{Tf,3}} = Vector{Array{Tf,3}}(undef,1)
    p̂_Θt::Vector{Matrix{Tf}} = Vector{Matrix{Tf}}(undef,1)
    X̂_Θt::Vector{Matrix{Tf}} = Vector{Matrix{Tf}}(undef,1)
    Ix̂_Θt::Vector{Matrix{Tf}} = Vector{Matrix{Tf}}(undef,1)
end


"""
    FirstOrderApproximation(ZO::ZerothOrderApproximation, T)

Constructs a first-order approximation object (inheriting ZO's precision Tf).
"""
function FirstOrderApproximation(ZO::ZerothOrderApproximation{Tf}, T) where {Tf}
    return FirstOrderApproximation{Tf}(ZO = ZO, T = T)
end


# ── precision-scaled tolerances and BB factorisation helpers ────────────────
_solve_rtol(::Type{Float64}) = 1e-6
_solve_rtol(::Type{Tf}) where {Tf<:AbstractFloat} = sqrt(eps(Tf))

"""
    _lu_dense_or_sparse(BB, nmax)

DENSE lu up to nmax (UMFPACK hits tiny pivots on BB's mixed row scales and
fails SILENTLY — see compute_BB!), sparse UMFPACK beyond (promoted to Float64).
"""
function _lu_dense_or_sparse(BB::SparseMatrixCSC{Tf,Int64}, nmax::Int) where {Tf}
    if size(BB, 1) <= nmax
        return _with_blas_threads(() -> lu(Matrix(BB)))
    else
        return lu(Tf === Float64 ? BB : SparseMatrixCSC{Float64,Int64}(BB))
    end
end

_bb_ldiv(luBB::Factorization{Tf}, v::AbstractVector{Tf}) where {Tf} = luBB \ v
_bb_ldiv(luBB::Factorization{Tl}, v::AbstractVector{Tf}) where {Tl,Tf} = Tf.(luBB \ Tl.(v))


"""
    reconstruct_rows!(out, Wzs, WXs, ẑ, QX̂)

Fill `out[r,j] = Σ_c Wzs[r,c,j]·ẑ[c,j] + Σ_q WXs[r,q,j]·QX̂[q]` — the selected
rows of the reconstruction x̂ = Wz ẑ + WX QX̂ at every node.
"""
function reconstruct_rows!(out::AbstractMatrix, Wzs::AbstractArray{<:Real,3}, WXs::AbstractArray{<:Real,3},
                           ẑ::AbstractMatrix, QX̂::AbstractVector)
    k, nz, nsp = size(Wzs)
    nQ = size(WXs, 2)
    threaded_foreach(nsp) do _, jr
        @inbounds for j in jr
            for r in 1:k
                s = zero(eltype(out))
                for c in 1:nz
                    s += Wzs[r,c,j]*ẑ[c,j]
                end
                for q in 1:nQ
                    s += WXs[r,q,j]*QX̂[q]
                end
                out[r,j] = s
            end
        end
    end
    return out
end


"""
    compute_f_matrices!(FO::FirstOrderApproximation)

Per-node blocks of the z-formulation.  With the z-equations/z-variables first,
    Fzz = dF.x[1:n.z,1:n.z]   Fzy = dF.x[1:n.z,n.z+1:end]     (etc.)
the intratemporal solution is y_z = −Fyy⁻¹Fyz, y_X = −Fyy⁻¹FyQ, and the
z-block Schur complement is
    D_j = Fzz + Fzy y_z + Fze·Eȳ_a[:,:,j]·p_z,       f_j = −D_j⁻¹,
with Eȳ_a = ∂E[ȳ′]/∂a′ the expected derivative of the taste-integrated
steady-state function (ȳ Φ̃ᵉₐ).  The full policy state-Jacobian is rebuilt
for the kink machinery:
    z̄_a = f_j (Fza + Fzy y_a),    x̄_a = [z̄_a; y_a + y_z z̄_a].
"""
function compute_f_matrices!(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO = FO
    @unpack ȳ, Φ̃ᵉₐ, dF, n, ie, iI = ZO
    nz, ny, ne, nye, nQ, na = n.z, n.y, n.e, n.ye, n.Q, n.a

    Eȳ_a = FO.Eȳ_a = reshape(ȳ*Φ̃ᵉₐ, nye, na, n.sp)

    FO.y_z    = zeros(Tf, ny, nz, n.sp)
    FO.y_X    = zeros(Tf, ny, nQ, n.sp)
    FO.invFyy = zeros(Tf, ny, ny, n.sp)
    FO.f      = zeros(Tf, nz, nz, n.sp)
    FO.fFze   = zeros(Tf, nz, nye, n.sp)
    FO.fF̃zQ  = zeros(Tf, nz, nQ, n.sp)
    FO.Wz     = zeros(Tf, n.x, nz, n.sp)
    FO.WX     = zeros(Tf, n.x, nQ, n.sp)
    FO.x̄_a   = zeros(Tf, n.x, na, n.sp)

    zi, yi = 1:nz, nz+1:n.x
    nt = n_tasks(n.sp)
    Dbufs = [zeros(Tf, nz, nz) for _ in 1:nt]
    threaded_foreach(n.sp, ntasks=nt) do chunk, jr
        Dbuf = Dbufs[chunk]
        for j in jr
            A  = dF.x[j]
            Ae = dF.xe[j]
            AX = dF.X[j]
            Aa = dF.a[j]
            Fzz = @view A[zi, zi];  Fzy = @view A[zi, yi]
            Fyz = @view A[yi, zi];  Fyy = @view A[yi, yi]
            Fze = @view Ae[zi, :]
            FzQ = @view AX[zi, :];  FyQ = @view AX[yi, :]
            Fza = @view Aa[zi, :];  Fya = @view Aa[yi, :]

            iFyy = inv(Matrix(Fyy))
            y_z = -iFyy*Fyz
            y_X = -iFyy*FyQ
            y_a = -iFyy*Fya

            # D = Fzz + Fzy y_z + Fze·Eȳ_a·p_z  (p_z = [I_na 0]: only the first
            # n.a columns of D get the expectation-shift term)
            Dbuf .= Fzz
            mul!(Dbuf, Fzy, y_z, one(Tf), one(Tf))
            @views mul!(Dbuf[:, 1:na], Fze, Eȳ_a[:,:,j], one(Tf), one(Tf))
            f_j = -inv(Dbuf)

            z̄_a = f_j*(Fza .+ Fzy*y_a)

            FO.invFyy[:,:,j] = iFyy
            FO.y_z[:,:,j]  = y_z
            FO.y_X[:,:,j]  = y_X
            FO.f[:,:,j]    = f_j
            FO.fFze[:,:,j] = f_j*Fze
            FO.fF̃zQ[:,:,j] = f_j*(FzQ .+ Fzy*y_X)
            FO.x̄_a[zi,:,j] = z̄_a
            FO.x̄_a[yi,:,j] = y_a .+ y_z*z̄_a
            for c in 1:nz
                FO.Wz[c,c,j] = one(Tf)
            end
            FO.Wz[yi,:,j] = y_z
            FO.WX[yi,:,j] = y_X
        end
    end
    FO.Sez = FO.Wz[ie,:,:]
    FO.SeX = FO.WX[ie,:,:]
    FO.IzW = FO.Wz[iI,:,:]
    FO.IXW = FO.WX[iI,:,:]
end


"""
    compute_Lemma3!(FO)

One pass through the z-kernel recursion
    z_0 = fF̃zQ,   z_1 = fFze·E_f[Sez z_0 + SeX],   z_s = fFze·E_f[Sez z_{s-1}],
storing ONLY the savings rows (FO.a), the choice-probability rows (FO.pk,
DC) and the ω̄-integrated Ilab rows (FO.Iz — row 1 of the J matrix, with
the direct ∫WX dω̄ term on the s=1 block and the Wp·pk term).  The kernels
themselves are never kept (paths come from the backward recursion in
compute_x̂t_Ω̂t!).
"""
function compute_Lemma3!(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO, T = FO
    @unpack n, IntΦ, Wp, iI = ZO
    @unpack fFze, fF̃zQ, Sez, SeX, IzW, IXW = FO
    WpI = Wp[iI, :]                                   # n.I × n.spr

    FO.a  = zeros(Tf, n.a, n.sp, n.Q, T)
    FO.pk = zeros(Tf, n.spr, n.Q, T)
    FO.Iz = zeros(Tf, n.I, n.Q, T)

    zs   = zeros(Tf, n.z, n.Q, n.sp)     # current kernel z_{s-1} (code s)
    xe   = zeros(Tf, n.e, n.Q, n.sp)     # its xelab rows (+SeX at s=1)
    ybuf = zeros(Tf, n.ye, n.Q, n.sp)
    Eye  = zeros(Tf, n.ye, n.Q, n.sp)
    nt = n_tasks(n.sp)
    Izbufs = [zeros(Tf, n.I, n.Q) for _ in 1:nt]

    zs .= fF̃zQ
    for s in 1:T
        # xelab rows of the current kernel (the s=1 kernel carries the direct
        # price response of next period's intratemporal variables, SeX)
        node_mul_sweep!(xe, Sez, zs, n.sp, ntasks=nt)
        s == 1 && (xe .+= SeX)
        # savings rows (states are z-rows 1:n.a)
        for d in 1:n.a
            @views FO.a[d,:,:,s] .= transpose(zs[d,:,:])
        end
        # (DC) choice-probability rows
        choice_prob!(view(FO.pk, :, :, s), xe, ZO)
        # integrated Ilab rows: Σ_j IntΦ[j] (IzW_j z_j + 1{s=1} IXW_j) + WpI·pk_s
        Iz_s = @view FO.Iz[:,:,s]
        threaded_foreach(n.sp, ntasks=nt) do chunk, jr
            Izc = Izbufs[chunk]
            fill!(Izc, zero(Tf))
            if nt == 1
                for j in jr
                    @views mul!(Izc, IzW[:,:,j], zs[:,:,j], IntΦ[j], one(Tf))
                end
            else
                for j in jr
                    @views _node_mul!(Izc, IzW[:,:,j], zs[:,:,j], IntΦ[j], one(Tf))
                end
            end
            if s == 1
                for j in jr
                    @views Izc .+= IntΦ[j] .* IXW[:,:,j]
                end
            end
        end
        for chunk in 1:nt
            Iz_s .+= Izbufs[chunk]
        end
        @views mul!(Iz_s, WpI, FO.pk[:,:,s], one(Tf), one(Tf))
        # next kernel
        if s < T
            expect_f!(Eye, xe, ybuf, ZO)
            node_mul_sweep!(zs, fFze, Eye, n.sp, ntasks=nt)
        end
    end
end


"""
    compute_Lemma4!(FO)

Operators of the lifted law of motion of the PRE-CHOICE distribution:
    L = Λ̃,   M[d,c] = D̄_d · Λc[c] · diag(ω̄_c) · Φrᵀ,   Mp = (Λc[1] − Λc[2]) · diag(μ̄) · Φrᵀ.
(The savings/probability kernels FO.a, FO.pk are filled by compute_Lemma3!.)
"""
function compute_Lemma4!(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO = FO
    @unpack Φr, ω̄, μ̄, Λ, Λc, n, D = ZO
    Φrt = transpose(Φr)

    FO.Lt = transpose(Λ)
    FO.L = Λ

    FO.M = Matrix{FactoredTransitionMatrix{Matrix{Tf},Tf}}(undef, n.a, n.c)
    for c in 1:n.c
        ω̄_c = diag_transition_matrix(ω̄[(c-1)*n.Ω+1:c*n.Ω], n.θ)
        for d in 1:n.a
            FO.M[d,c] = D[d] * Λc[c] * ω̄_c * Φrt
        end
    end
    FO.Mt = [transpose(M) for M in FO.M]

    # (DC) mass reassigned between occupations by a change in Prob(c = 1)
    μ̄_diag = diag_transition_matrix(μ̄, n.θ)
    Φs = [(Λc[1].Φs[s] .- Λc[2].Φs[s]) * μ̄_diag.Φs[s] * Φrt.Φs[s] for s in 1:n.θ]
    FO.Mp = FactoredTransitionMatrix(Φs=Φs, Π=Matrix{Tf}(Λc[1].Π))
    FO.Mpt = transpose(FO.Mp)
end


"""
    compute_Corollary2!(FO)

Builds the streamed distribution-propagation stack (Z idea p.3) and, from it,
the IA tensor.  ONE forward Λ̃-stream over the TRANSPOSED operators fills
    ILMt[k,i,t] = [Iμ Λ̃^{t-1} M_k]_{i,·},
with k running over the flattened savings index (d,j) — d fastest, j the FULL
node with c slowest — followed by the reduced node index of Mp, so that
    IA[(i,t),(q,s)] = Σ_k ILMt[k,i,t] · K[k,q,s],   K = [a; pk]
is a SINGLE gemm over the stacked kernels.

Requires compute_Lemma3! (fills FO.a, FO.pk) and compute_Lemma4! (fills Lt/Mt/Mpt).
"""
function compute_Corollary2!(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO, T, Mt, Mpt, Lt = FO
    @unpack n, iI, Iμ = ZO
    nK = n.a*n.sp + n.spr

    IL = Iμ[iI, :]                                 # n.I × n.Ω  (values against μ̂)
    FO.ILMt = ILMt = zeros(Tf, nK, n.I, T-1)
    IΛt  = copy(IL')
    IΛ2t = copy(IΛt)
    tmp  = zeros(Tf, n.spr, n.I)
    tmpa = zeros(Tf, n.a, n.sp, n.I)
    for t in 1:T-1
        for c in 1:n.c, d in 1:n.a
            @doinplace tmp .= Mt[d,c]*IΛt
            @views tmpa[d, (c-1)*n.spr+1:c*n.spr, :] .= tmp
        end
        @views ILMt[1:n.a*n.sp, :, t] .= reshape(tmpa, n.a*n.sp, n.I)
        @doinplace tmp .= Mpt*IΛt
        @views ILMt[n.a*n.sp+1:end, :, t] .= tmp
        if t < T-1
            @doinplace IΛ2t .= Lt*IΛt
            IΛt, IΛ2t = IΛ2t, IΛt
        end
    end
    K = zeros(Tf, nK, n.Q, T)
    K[1:n.a*n.sp, :, :] .= reshape(FO.a, n.a*n.sp, n.Q, T)
    K[n.a*n.sp+1:end, :, :] .= FO.pk
    IA = _with_blas_threads() do
        reshape(ILMt, nK, n.I*(T-1))'*reshape(K, nK, n.Q*T)
    end
    FO.IA = reshape(IA, n.I, T-1, n.Q, T)
end


"""
    compute_Proposition1!(FO)

The J recursion.  Row 1 of J is the kernel-integral block FO.Iz; the rest
accumulates FO.IA, which compute_Corollary2! must have built first.
"""
function compute_Proposition1!(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO, T, IA, Iz = FO
    @unpack n = ZO

    J = FO.J = zeros(Tf, n.I, T, n.Q, T)
    J[:,1,:,:] .= Iz
    @views for t in 2:T
        J[:,t,:,1] .= IA[:,t-1,:,1]
    end
    @views for s in 2:T
        for t in 2:T
            J[:,t,:,s] .= J[:,t-1,:,s-1] .+ IA[:,t-1,:,s]
        end
    end
end


"""
    compute_BB!(FO::FirstOrderApproximation)

Computes the BB matrix (Proposition 1).  dG.x is restricted to the Ilab
columns (all others are zero — verified by computeDerivativesG!).
"""
function compute_BB!(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO, T, J = FO
    @unpack dG, P, Q, n, iI = ZO
    dGx_I = dG.x[:, iI]
    ITT  = sparse(I,T,T)
    ITT_ = spdiagm(-1=>ones(Tf,T-1))
    ITTᵉ = spdiagm(1=>ones(Tf,T-1))
    FO.BB = kron(ITT,dGx_I)*reshape(J,n.I*T,:)*kron(ITT,Q) .+ kron(ITT,dG.X) .+
            kron(ITTᵉ,dG.Xᵉ) .+ kron(ITT_,dG.X_*P)
    # DENSE factorisation, deliberately: UMFPACK hits tiny pivots on BB's
    # mixed row scales and fails SILENTLY (see the MPC code); dense up to
    # 50k rows is cheap next to a wrong answer.
    FO.luBB = _lu_dense_or_sparse(FO.BB, 50_000)
end


"""
    solve_Xt!(FO::FirstOrderApproximation)

Solves for the path X̂t (aggregate variable responses) given the initial
conditions Θ_0, Ω_0 (= μ̂_1 on the reduced fine grid) and X_0.
"""
function solve_Xt!(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO, T, Θ_0, Ω_0, X_0, luBB = FO
    @unpack dG, n, Λ, ρ_Θ, iI, Iμ = ZO
    IL = Iμ[iI, :]
    dGx_I = dG.x[:, iI]
    AA = zeros(Tf, n.X,T)
    for t in 1:T
        @views AA[:,t] .+= dG.Θ*ρ_Θ^(t-1)*Θ_0
    end

    if length(Ω_0) == n.Ω && any(!iszero, Ω_0)
        Ωt = zeros(Tf, n.Ω)
        Ωt_ = copy(Ω_0)
        for t in 1:T
            AA[:,t] .+= dGx_I*(IL*Ωt_)
            @doinplace Ωt .= Λ*Ωt_
            Ωt, Ωt_ = Ωt_, Ωt
        end
    end

    AA[:,1] .+= dG.X_*X_0

    Xt = -_bb_ldiv(luBB, AA[:])
    let r = norm(FO.BB*Xt .+ AA[:]) / max(norm(AA[:]), eps(Tf))
        r > _solve_rtol(Tf) && @warn "solve_Xt!: BB solve residual $(r) — factorisation is unreliable"
    end
    FO.X̂t = reshape(Xt,n.X,T)
end


"""
    compute_x̂t_Ω̂t!(FO::FirstOrderApproximation)

Computes ẑt (the tracked policy perturbations) by the BACKWARD recursion
    ẑ_t = fFze·E_f[Sez ẑ_{t+1} + SeX QX̂_{t+1}] + fF̃zQ·QX̂_t,   ẑ_{T+1} = 0,
the choice-probability path p̂t (DC), then pushes the PRE-CHOICE distribution
forward with
    μ̂_{t+1} = Λ̃ μ̂_t + Σ_{d,c} M[d,c]·ẑ_t[d, (·,c)] + Mp·p̂_t,   μ̂_1 = Ω_0.
"""
function compute_x̂t_Ω̂t!(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO, T, X̂t, L, M, Mp, Ω_0, fFze, fF̃zQ, Sez, SeX = FO
    @unpack Q, n = ZO
    QX̂ = Q*X̂t                             # n.Q × T
    ẑt = FO.ẑt = zeros(Tf, n.z, n.sp, T)
    p̂t = FO.p̂t = zeros(Tf, n.spr, T)
    xe   = zeros(Tf, n.e, n.sp)
    ybuf = zeros(Tf, n.ye, n.sp)
    Eye  = zeros(Tf, n.ye, n.sp)

    for t in T:-1:1
        if t < T
            reconstruct_rows!(xe, Sez, SeX, view(ẑt,:,:,t+1), view(QX̂,:,t+1))
            expect_f!(Eye, xe, ybuf, ZO)
        else
            Eye .= zero(Tf)
        end
        QX̂_t = @view QX̂[:,t]
        threaded_foreach(n.sp) do _, jr
            for j in jr
                z_j = @view ẑt[:,j,t]
                @views mul!(z_j, fFze[:,:,j], Eye[:,j])
                @views mul!(z_j, fF̃zQ[:,:,j], QX̂_t, one(Tf), one(Tf))
            end
        end
    end

    # choice-probability path from the xelab rows of x̂_t
    for t in 1:T
        reconstruct_rows!(xe, Sez, SeX, view(ẑt,:,:,t), view(QX̂,:,t))
        choice_prob!(view(p̂t, :, t), xe, ZO)
    end

    # push the pre-choice distribution forward
    Ω̂t = FO.Ω̂t = zeros(Tf, n.Ω,T)
    if length(Ω_0) == n.Ω
        Ω̂t[:,1] .= Ω_0
    end
    âbuf = zeros(Tf, n.spr)
    for t in 2:T
        Ω̂t_t_ = @view(Ω̂t[:,t-1])
        Ω̂t_t  = @view(Ω̂t[:,t])
        @doinplace Ω̂t_t .= L * Ω̂t_t_
        for c in 1:n.c, d in 1:n.a
            âbuf .= @view ẑt[d, (c-1)*n.spr+1:c*n.spr, t-1]
            @doinplace Ω̂t_t .+= M[d,c] * âbuf
        end
        p̂_t_ = @view p̂t[:,t-1]
        @doinplace Ω̂t_t .+= Mp * p̂_t_
    end
end


"""
    compute_Ixt(FO::FirstOrderApproximation)

The path of the integrated individual variables,
    Ix̂_t = x̂_t·IntΦ + Wp p̂_t + Iμ μ̂_t          (n.x × T),
with x̂_t reconstructed from ẑt and X̂t.
"""
function compute_Ixt(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO, ẑt, p̂t, Ω̂t, X̂t, T, Wz, WX = FO
    @unpack n, IntΦ, Wp, Iμ, Q = ZO
    QX̂ = Q*X̂t
    Ixt = zeros(Tf, n.x, T)
    x̂buf = zeros(Tf, n.x, n.sp)
    for t in 1:T
        reconstruct_rows!(x̂buf, Wz, WX, view(ẑt,:,:,t), view(QX̂,:,t))
        @views Ixt[:,t] .= x̂buf*IntΦ .+ Wp*p̂t[:,t] .+ Iμ*Ω̂t[:,t]
    end
    return Ixt
end


"""
    compute_Θ_derivatives!(FO)

Computes the derivatives in each Θ direction (builds the whole kernel/J/BB
chain, then solves once per aggregate shock).
"""
function compute_Θ_derivatives!(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO = FO
    @unpack n = ZO

    FO.X̂_Θt = Vector{Matrix{Tf}}(undef,n.Θ)
    FO.X_0 = zeros(Tf, n.A)
    FO.Ω_0 = zeros(Tf, n.Ω)
    compute_f_matrices!(FO)
    compute_Lemma3!(FO)
    compute_Lemma4!(FO)
    compute_Corollary2!(FO)
    compute_Proposition1!(FO)
    compute_BB!(FO)
    for i in 1:n.Θ
        FO.Θ_0 = I[1:n.Θ,i]
        solve_Xt!(FO)
        FO.X̂_Θt[i] = FO.X̂t
    end
end


"""
    compute_x_Θ_derivatives!(FO)

Individual/distribution paths for each Θ direction (after compute_Θ_derivatives!).
"""
function compute_x_Θ_derivatives!(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO, T = FO
    @unpack n = ZO

    FO.Ω̂_Θt = Vector{Matrix{Tf}}(undef,n.Θ)
    FO.ẑ_Θt = Vector{Array{Tf,3}}(undef,n.Θ)
    FO.p̂_Θt = Vector{Matrix{Tf}}(undef,n.Θ)
    FO.Ix̂_Θt = Vector{Matrix{Tf}}(undef,n.Θ)
    FO.X_0 = zeros(Tf, n.A)
    FO.Ω_0 = zeros(Tf, n.Ω)
    for i in 1:n.Θ
        FO.Θ_0 = I[1:n.Θ,i]
        FO.X̂t = FO.X̂_Θt[i]
        compute_x̂t_Ω̂t!(FO)
        FO.Ω̂_Θt[i] = FO.Ω̂t
        FO.ẑ_Θt[i] = FO.ẑt
        FO.p̂_Θt[i] = FO.p̂t
        FO.Ix̂_Θt[i] = compute_Ixt(FO)
    end
end


"""
    check_J_consistency(FO)

Internal consistency diagnostic: the integrated Ilab paths computed DIRECTLY
(backward recursion + distribution push-forward, `compute_Ixt`) must equal
the J-matrix representation
    Ix̂_t[iI] = Σ_s J[:,t,:,s] QX̂_s + Iμ Λ̃^{t-1} Ω_0.
Returns the maximum absolute discrepancy (relative to the path scale).
Requires solve_Xt! and compute_x̂t_Ω̂t! to have been run.
"""
function check_J_consistency(FO::FirstOrderApproximation{Tf}) where {Tf}
    @unpack ZO, T, J, X̂t, Ω_0 = FO
    @unpack n, iI, Q, Λ, Iμ = ZO
    Ixt = compute_Ixt(FO)[iI, :]
    QX̂ = Q*X̂t
    IxJ = reshape(reshape(J, n.I*T, n.Q*T)*QX̂[:], n.I, T)
    if length(Ω_0) == n.Ω
        IL = Iμ[iI, :]
        Ωt_ = copy(Ω_0); Ωt = similar(Ωt_)
        for t in 1:T
            IxJ[:,t] .+= IL*Ωt_
            @doinplace Ωt .= Λ*Ωt_
            Ωt, Ωt_ = Ωt_, Ωt
        end
    end
    return maximum(abs, Ixt .- IxJ) / max(maximum(abs, Ixt), eps(Tf))
end
