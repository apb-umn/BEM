"""
    PCHIP.jl

Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation matrices
and FactoredBasisMatrix storage for block-diagonal basis matrices.

Exports:
  - pchip_slope_matrix(x)       → sparse N×N matrix D s.t. d = D*y gives slopes
  - pchip_eval_matrix(x_from, x_to)  → sparse matrix mapping values → interpolated values
  - pchip_deriv_matrix(x_from, x_to) → sparse matrix mapping values → interpolated derivatives
  - FactoredBasisMatrix         → block-diagonal storage with exogenous transition
  - LUFactoredBasisMatrix       → LU factorizations of per-θ blocks
"""

using SparseArrays, LinearAlgebra


# ============================================================
# Threading helpers (added 2026-08-18)
#
# The hot loops of the perturbation code — the per-node (n.sp) loops of the
# First/Second order files and the per-θ-block loops of the factored-operator
# multiplies below — are embarrassingly parallel: every iteration writes a
# DISJOINT slice of the output.  They all run through `threaded_foreach`,
# which partitions the index range into contiguous chunks and spawns one task
# per chunk.  Conventions:
#   * Parallelism is controlled ONLY by the Julia session's thread count
#     (`julia -t N`); with 1 thread every call takes the exact legacy serial
#     path (`body!(1, 1:n)`), so single-threaded runs remain BITWISE
#     identical to the pre-threading code.
#   * The chunk id `c` passed to `body!` indexes per-chunk scratch buffers at
#     the call sites (the low-allocation buffer style is kept by REPLICATING
#     each small per-node buffer once per chunk, not per iteration).  Two
#     tasks never share a chunk id, and the partition depends only on
#     (n, ntasks) — never on runtime scheduling.
#   * Reduction loops (the few Σ_j accumulations) keep one PARTIAL per chunk,
#     combined in chunk order after the sync — deterministic for a fixed
#     thread count, and exactly the legacy sum when ntasks == 1.
#   * A threaded run may differ from a serial one only in (a) those chunked
#     reductions (floating-point reassociation) and (b) the per-node
#     (n.z × n.Q)-shaped products, which switch from BLAS to the plain-loop
#     `_node_mul!` kernel when threaded (see the gemm-lock note below;
#     last-ulp fma differences).  All other disjoint-write loops are bitwise
#     identical at any thread count, and `-t 1` is always bitwise legacy.
# ============================================================

const THREAD_MIN_CHUNK = 1024   # minimum per-task slice of a per-node loop

"""
    n_tasks(n; min_chunk=THREAD_MIN_CHUNK)

Number of chunks a hot loop over `1:n` is split into: 1 in a single-threaded
session (or when `n` is too small to be worth spawning for), otherwise up to
4× the thread count (the oversubscription keeps heterogeneous cores busy).
Depends only on `n` and the session's thread count, so results are
reproducible for a fixed `julia -t N`.  Loops whose per-iteration work is
large (e.g. over time periods rather than grid nodes) pass `min_chunk=1`.
"""
n_tasks(n::Int; min_chunk::Int=THREAD_MIN_CHUNK) =
    Threads.nthreads() == 1 ? 1 : clamp(n ÷ min_chunk, 1, 4*Threads.nthreads())

"""
    threaded_foreach(body!, n; ntasks=n_tasks(n))

Run `body!(c, jr)` over a deterministic partition of `1:n` into `ntasks`
contiguous chunks; `c` is the chunk index (use it to pick per-chunk scratch
buffers — allocate them as `[make_buf() for _ in 1:ntasks]` with the SAME
`ntasks` value).  With `ntasks == 1` this is exactly `body!(1, 1:n)` — no
task is spawned and the serial code path is untouched.
"""
function threaded_foreach(body!::B, n::Int; ntasks::Int=n_tasks(n)) where {B}
    if ntasks <= 1 || n == 0
        n > 0 && body!(1, 1:n)
        return nothing
    end
    @sync for c in 1:ntasks
        jr = (div((c-1)*n, ntasks)+1):div(c*n, ntasks)
        Threads.@spawn body!(c, jr)
    end
    return nothing
end

#  per-θ-block loops of the factored operators: one task per block bundle,
#  only when the operator is big enough for the spawn overhead to vanish
_ftm_ntasks(S::Int, work::Int) =
    (Threads.nthreads() == 1 || work < 32_768) ? 1 : min(Threads.nthreads(), S)

# ── tiny per-node matmuls under threading ───────────────────────────────────
# OpenBLAS's gemm takes a GLOBAL pack-buffer lock, so concurrent tiny gemms
# from many Julia threads SERIALIZE and collapse — measured 3× slower than
# serial for the (n.z × n.Q)-shaped per-node products (gemv and the sparse
# kernels don't touch that buffer and scale fine).  The threaded branches of
# the per-node sweeps therefore use the plain-loop kernels below instead of
# BLAS; the SERIAL branches keep the legacy `mul!` call, so `julia -t 1`
# stays bitwise-identical to the pre-threading code.  Threaded results can
# differ from serial in these products only at the last-ulp level (separate
# mul/add vs BLAS's fused-multiply-add) — same class as the chunked-reduction
# reassociation.

@inline function _node_mul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    m, k = size(A)
    n = size(B, 2)
    @inbounds for q in 1:n, r in 1:m
        s = zero(eltype(C))
        for c in 1:k
            s += A[r,c]*B[c,q]
        end
        C[r,q] = s
    end
    return C
end

#  C = α·A·B + β·C (the accumulate form of the reduction/rhs sites)
@inline function _node_mul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix,
                            α::Real, β::Real)
    m, k = size(A)
    n = size(B, 2)
    @inbounds for q in 1:n, r in 1:m
        s = zero(eltype(C))
        for c in 1:k
            s += A[r,c]*B[c,q]
        end
        C[r,q] = α*s + β*C[r,q]
    end
    return C
end

"""
    node_mul_sweep!(out, A, B, n; ntasks=n_tasks(n))

`out[:,:,j] = A[:,:,j] * B[:,:,j]` for `j in 1:n` — the recurring per-node
sweep of the kernel recursions.  Serial (`ntasks == 1`): the legacy BLAS
`mul!` loop, bitwise the pre-threading code.  Threaded: `_node_mul!` per
node (see the OpenBLAS gemm-lock note above).
"""
function node_mul_sweep!(out::AbstractArray{<:Real,3}, A::AbstractArray{<:Real,3},
                         B::AbstractArray{<:Real,3}, n::Int; ntasks::Int=n_tasks(n))
    if ntasks <= 1
        for j in 1:n
            @views mul!(out[:,:,j], A[:,:,j], B[:,:,j])
        end
    else
        threaded_foreach(n, ntasks=ntasks) do _, jr
            for j in jr
                @views _node_mul!(out[:,:,j], A[:,:,j], B[:,:,j])
            end
        end
    end
    return out
end


# ============================================================
# FactoredBasisMatrix: block-diagonal storage for basis matrices
#
# Decomposes an n.sp × n.sp basis matrix into per-theta-state
# endogenous blocks plus an exogenous transition matrix.
# For a vector x of length n.sp = na * n.θ:
#   1. Reshape x into (na, n.θ) columns
#   2. Multiply each column s by its block Φs[s] (endogenous interpolation)
#   3. Multiply the result by Π (exogenous mixing across theta states)
# ============================================================

"""
    FactoredTransitionMatrix{T,Tv}

Block-diagonal transition matrix with exogenous transition.
- `Φs`: Vector of per-theta-state sparse matrices, each na_out × na_in
- `Π`: Exogenous transition matrix (or I for block-diagonal)
- `xbuffer`: Pre-allocated buffer for intermediate results

`Tv` is the floating-point element type of the blocks (Float64 by default;
Float32/Float16 for reduced-precision runs — see `_to_precision`).  A partial
annotation `FactoredTransitionMatrix{Matrix{Float64}}` (element type free)
remains valid wherever the old one-parameter form was used.

Φs represent the endogenous transition matrices and Π represents the exogenous transition matrix.
Let i represent the endogenous gridpoints and s represent the exogenous gridpoints.
Then we are implimenting the transition
    y'[i',s'] = sum_{s,i} Φs[s][i',i] y[i,s] Π[s',s]
"""
@kwdef mutable struct FactoredTransitionMatrix{T,Tv<:AbstractFloat}
    Φs::Vector{SparseMatrixCSC{Tv,Int64}} = [spzeros(1,1) for _ in 1:1]
    Π::T

    # Buffer for intermediate results (size = na_in * n.θ)
    xbuffer::Vector{Tv} = zeros(eltype(Φs[1]), size(Φs[1],1)*length(Φs))
end

"""
    FactoredBasisMatrixTransposed{T,Tv}

Transpose of a Block-diagonal transition matrix with exogenous transition. To
perform expectation operators
- `Φs`: Vector of per-theta-state transposed sparse matrices Φs[s][i,i']
- `Π`: Exogenous transition matrix (or I for block-diagonal) Π[s,s']
- `xbuffer`: Pre-allocated buffer for intermediate results

This allows us to implement the conditional expectation operator
     Ey[i,s] = sum_{s',i'} Φs[s][i,i'] y[i',s'] Π[s,s']
"""
@kwdef mutable struct FactoredTransitionMatrixTransposed{T,Tv<:AbstractFloat}
    Φs::Vector{SparseMatrixCSC{Tv,Int64}} = [spzeros(1,1) for _ in 1:1]
    Π::T

    # Buffer for intermediate results (size = na_in * n.θ)
    xbuffer::Vector{Tv} = zeros(eltype(Φs[1]), size(Φs[1],2)*length(Φs))
end

# ---- element type & precision conversion ----

"""
    _to_precision(Tv, A)

Convert the floating-point element type of an operator/array to `Tv`:
sparse and dense arrays elementwise; `FactoredTransitionMatrix`(/`Transposed`)
blockwise (Π included when it is a matrix); `UniformScaling` untouched.  Used
by the reduced-precision (Float32/Float16) builds of the approximation code.
"""
_to_precision(::Type{Tv}, A::SparseMatrixCSC) where {Tv<:AbstractFloat} =
    SparseMatrixCSC{Tv,Int64}(A)
_to_precision(::Type{Tv}, A::AbstractMatrix) where {Tv<:AbstractFloat} = Matrix{Tv}(A)
_to_precision(::Type{Tv}, A::AbstractVector) where {Tv<:AbstractFloat} = Vector{Tv}(A)
_to_precision(::Type{Tv}, A::UniformScaling) where {Tv<:AbstractFloat} = A
_to_precision(::Type{Tv}, Φ::FactoredTransitionMatrix) where {Tv<:AbstractFloat} =
    FactoredTransitionMatrix(Φs=[_to_precision(Tv, B) for B in Φ.Φs],
                             Π=_to_precision(Tv, Φ.Π))
_to_precision(::Type{Tv}, Φ::FactoredTransitionMatrixTransposed) where {Tv<:AbstractFloat} =
    FactoredTransitionMatrixTransposed(Φs=[_to_precision(Tv, B) for B in Φ.Φs],
                                       Π=_to_precision(Tv, Φ.Π))

_ftm_eltype(Φ::FactoredTransitionMatrix{T,Tv}) where {T,Tv} = Tv
_ftm_eltype(Φ::FactoredTransitionMatrixTransposed{T,Tv}) where {T,Tv} = Tv

#  convert methods so typed struct fields (e.g. Inputs{Tf}.Λ) accept a
#  factored matrix of any precision; the ::annotation guards Π-type mismatch
Base.convert(::Type{FactoredTransitionMatrix{T,Tv}}, Φ::FactoredTransitionMatrix) where {T,Tv<:AbstractFloat} =
    _to_precision(Tv, Φ)::FactoredTransitionMatrix{T,Tv}
Base.convert(::Type{FactoredTransitionMatrixTransposed{T,Tv}}, Φ::FactoredTransitionMatrixTransposed) where {T,Tv<:AbstractFloat} =
    _to_precision(Tv, Φ)::FactoredTransitionMatrixTransposed{T,Tv}

# --- diag_transition_matrix ---
function diag_transition_matrix(ω::AbstractVector,S::Int64)
    ω = reshape(ω, :, S)
    Φs = [spdiagm(ω[:,s]) for s in 1:S]
    return FactoredTransitionMatrix(Φs=Φs, Π=I)
end


# ---- Custom mul! for FactoredTransitionMatrix ----

"""
    mul!(xout, Φ::FactoredTransitionMatrix, x)

In-place multiply: xout = Φ * x.
For vectors: x is length na_in * S, xout is length na_out * S.
Steps 
  1. Transition Endogenous States: xbuf[:,s] = Φs[s] * xbuf[:,s]
  2. Transition Exogenous States: xout[:,s] = xbuf * Π'
"""
function LinearAlgebra.mul!(xout::AbstractVector, Φ::FactoredTransitionMatrix, x::AbstractVector)
    #setup
    S = length(Φ.Φs)
    na_in = size(Φ.Φs[1], 2)
    na_out = size(Φ.Φs[1], 1)
    xr = reshape(x, :, S)
    if length(Φ.xbuffer) != na_out * S
        resize!(Φ.xbuffer, na_out * S)
    end
    xbufr = reshape(Φ.xbuffer, :, S)

    #apply steps (θ-blocks write disjoint buffer columns — threaded when big)
    threaded_foreach(S, ntasks=_ftm_ntasks(S, length(xout))) do _, sr
        for s in sr
            mul!(view(xbufr, :, s), Φ.Φs[s], view(xr, :, s))
        end
    end
    xoutr = reshape(xout, na_out, S)
    mul!(xoutr,xbufr, Φ.Π') # mix across θ states.... works for general Π and I
    return xout
end


"""
    mul!(xout, x , Φ::FactoredTransitionMatrix)

In-place multiply: xout = x * Φ .
For Arrays: x last dimension is length n_in * S, xout is length n_out * S.
reshape x into (:,n_in, S) and xout into (:,n_out, S)
allocate xbuffer of size x if it is not the right size
Steps 
  1. Mix across θ states: xbuf = x * Π
  2. Transition Endogenous States: xbuf[:,s] = Φs[s] * xbuf[:,s]
"""
function LinearAlgebra.mul!(xout::AbstractArray, x::AbstractArray, Φ::FactoredTransitionMatrix)
    #get sizes
    S = length(Φ.Φs)
    na_in = size(Φ.Φs[1], 1)
    na_out = size(Φ.Φs[1], 2)
    #make sure xbuffer is the right size
    if length(Φ.xbuffer) != length(x)
        resize!(Φ.xbuffer, length(x))
    end
    #setup
    xbufr = reshape(Φ.xbuffer, :, S)
    xr = reshape(x, :, S)
    #apply step1: Mix across θ states: xbuf = x * Π
    mul!(xbufr, xr, Φ.Π) #take expectations over \theta states
    #apply step2: (disjoint output slabs per θ-block — threaded when big)
    xbufr2 = reshape(xbufr,:,na_in,S)
    xoutr = reshape(xout,:,na_out,S)
    threaded_foreach(S, ntasks=_ftm_ntasks(S, length(xout))) do _, sr
        for s in sr
            mul!(view(xoutr,:,:,s),view(xbufr2,:,:,s),Φ.Φs[s])
        end
    end
    return xout
end


"""
    *(X::AbstractArray, Φ::FactoredTransitionMatrix)

Allocating multiply: returns Φ * X for matrices.
"""
function Base.:*(X::AbstractArray, Φ::FactoredTransitionMatrix)
    na_out = size(Φ.Φs[1], 2)
    S = length(Φ.Φs)
    Xoutsize = size(X)[1:end-1]...,na_out * S
    Xout = zeros(promote_type(eltype(X), _ftm_eltype(Φ)), Xoutsize)
    mul!(Xout, X, Φ)
    return Xout
end


"""
    mul!(xout, Φ::FactoredTransitionMatrix, x, α, β)

In-place multiply: xout = α * Φ * x + β * xout.
For vectors: x is length na_in * S, xout is length na_out * S.
Steps 
  1. Transition Endogenous States: xbuf[:,s] = Φs[s] * xbuf[:,s]
  2. Transition Exogenous States: xout[:,s] = xbuf * Π'
"""
function LinearAlgebra.mul!(xout::AbstractVector, Φ::FactoredTransitionMatrix, x::AbstractVector, α::Real, β::Real)
    #setup
    S = length(Φ.Φs)
    na_in = size(Φ.Φs[1], 2)
    na_out = size(Φ.Φs[1], 1)
    xr = reshape(x, :, S)
    if length(Φ.xbuffer) != na_out * S   # another orientation may have resized it
        resize!(Φ.xbuffer, na_out * S)
    end
    xbufr = reshape(Φ.xbuffer, :, S)

    #apply steps (θ-blocks write disjoint buffer columns — threaded when big)
    threaded_foreach(S, ntasks=_ftm_ntasks(S, length(xout))) do _, sr
        for s in sr
            mul!(view(xbufr, :, s), Φ.Φs[s], view(xr, :, s))
        end
    end
    xoutr = reshape(xout, na_out, S)
    mul!(xoutr,xbufr, Φ.Π', α, β) # mix across θ states.... works for general Π and I
    return xout
end

"""
    mul!(Xout, Φ::FactoredTransitionMatrix, X)

In-place multiply for matrices: Xout = Φ * X, column by column.
(The column loop must stay SERIAL — every column shares Φ.xbuffer; the
vector method threads its θ-block loop internally.)
"""
function LinearAlgebra.mul!(Xout::AbstractMatrix, Φ::FactoredTransitionMatrix, X::AbstractMatrix)
    for j in axes(X, 2)
        mul!(view(Xout, :, j), Φ, view(X, :, j))
    end
    return Xout
end

"""
    mul!(Xout, Φ::FactoredTransitionMatrix, X, α, β)

In-place multiply for matrices: Xout = Φ * X, column by column.
"""
function LinearAlgebra.mul!(Xout::AbstractMatrix, Φ::FactoredTransitionMatrix, X::AbstractMatrix, α::Real, β::Real)
    for j in axes(X, 2)
        mul!(view(Xout, :, j), Φ, view(X, :, j), α, β)
    end
    return Xout
end

"""
    *(Φ::FactoredTransitionMatrix, x::AbstractVector)

Allocating multiply: returns Φ * x.
"""
function Base.:*(Φ::FactoredTransitionMatrix, x::AbstractVector)
    na_out = size(Φ.Φs[1], 1)
    S = length(Φ.Φs)
    xout = Vector{promote_type(eltype(x), _ftm_eltype(Φ))}(undef, na_out * S)
    mul!(xout, Φ, x)
    return xout
end

"""
    *(Φ::FactoredTransitionMatrix, X::AbstractMatrix)

Allocating multiply: returns Φ * X for matrices.
"""
function Base.:*(Φ::FactoredTransitionMatrix, X::AbstractMatrix)
    na_out = size(Φ.Φs[1], 1)
    S = length(Φ.Φs)
    Xout = Matrix{promote_type(eltype(X), _ftm_eltype(Φ))}(undef, na_out * S, size(X, 2))
    mul!(Xout, Φ, X)
    return Xout
end


"""
    *(Φ1::FactoredTransitionMatrix, Φ2::FactoredTransitionMatrix)

Allocating multiply for FactoredTransitionMatrix: returns Φ1 * Φ2 for matrices.
"""
function Base.:*(Φ1::FactoredTransitionMatrix, Φ2::FactoredTransitionMatrix)
    S = length(Φ1.Φs)
    Φout = FactoredTransitionMatrix(
        Φs=[Φ1.Φs[s] * Φ2.Φs[s] for s in 1:S],
        Π=Φ1.Π * Φ2.Π)
    return Φout
end

function LinearAlgebra.transpose(Φ::FactoredTransitionMatrix{UniformScaling{Bool}})
    S = length(Φ.Φs)
    Φout = FactoredTransitionMatrix(
        Φs=[sparse(transpose(Φ.Φs[s])) for s in 1:S],
        Π=I)
    return Φout
end

function LinearAlgebra.transpose(Φ::FactoredTransitionMatrix)
    S = length(Φ.Φs)
    Φout = FactoredTransitionMatrixTransposed(
            Φs=[sparse(transpose(Φ.Φs[s])) for s in 1:S],
            Π=Matrix(Φ.Π'))
    return Φout
end

function LinearAlgebra.transpose(Φ::FactoredTransitionMatrixTransposed)
    S = length(Φ.Φs)
    if Φ.Π isa UniformScaling
        Φout = FactoredTransitionMatrix(
            Φs=[sparse(transpose(Φ.Φs[s])) for s in 1:S],
            Π=I)
    else
        Φout = FactoredTransitionMatrix(
            Φs=[sparse(transpose(Φ.Φs[s])) for s in 1:S],
            Π=Matrix(Φ.Π'))
    end
    return Φout
end

"""
    mul!(xout, Φ::FactoredTransitionMatrixTransposed, x)

In-place multiply: xout = Φ * x.
For vectors: x is length na_in * S, xout is length na_out * S.
Steps (because we are taking a conditional expectation and the Φ[s] are conditional on the current state s):
  1. Mix across θ states: xbuf = x * Π
  2. Multiply each θ-block: xout[:,s] = Φs[s] * xbuf[:,s]
"""
function LinearAlgebra.mul!(xout::AbstractVector, Φ::FactoredTransitionMatrixTransposed, x::AbstractVector)
    #setup
    S = length(Φ.Φs)
    na_in = size(Φ.Φs[1], 2)
    na_out = size(Φ.Φs[1], 1)
    xr = reshape(x, :, S)
    if length(Φ.xbuffer) != length(x)    # another orientation may have resized it
        resize!(Φ.xbuffer, length(x))
    end
    xbuf = reshape(Φ.xbuffer, :, S)

    #apply steps
    mul!(xbuf,xr, Φ.Π') # mix across θ states.... works for general Π and I
    xoutr = reshape(xout, na_out, S)
    #  θ-blocks write disjoint output columns — threaded when big
    threaded_foreach(S, ntasks=_ftm_ntasks(S, length(xout))) do _, sr
        for s in sr
            mul!(view(xoutr, :, s), Φ.Φs[s], view(xbuf, :, s))
        end
    end
    return xout
end


"""
    mul!(Xout, Φ::FactoredTransitionMatrixTransposed, X)

In-place multiply for matrices: Xout = Φ * X, column by column.
(Column loop serial — shared Φ.xbuffer; see the FactoredTransitionMatrix
matrix method.)
"""
function LinearAlgebra.mul!(Xout::AbstractMatrix, Φ::FactoredTransitionMatrixTransposed, X::AbstractMatrix)
    for j in axes(X, 2)
        mul!(view(Xout, :, j), Φ, view(X, :, j))
    end
    return Xout
end

"""
    *(Φ::FactoredTransitionMatrixTransposed, x::AbstractVector)

Allocating multiply: returns Φ * x.
"""
function Base.:*(Φ::FactoredTransitionMatrixTransposed, x::AbstractVector)
    na_out = size(Φ.Φs[1], 1)
    S = length(Φ.Φs)
    xout = Vector{promote_type(eltype(x), _ftm_eltype(Φ))}(undef, na_out * S)
    mul!(xout, Φ, x)
    return xout
end

"""
    *(Φ::FactoredTransitionMatrixTransposed, X::AbstractMatrix)

Allocating multiply: returns Φ * X for matrices.
"""
function Base.:*(Φ::FactoredTransitionMatrixTransposed, X::AbstractMatrix)
    na_out = size(Φ.Φs[1], 1)
    S = length(Φ.Φs)
    Xout = Matrix{promote_type(eltype(X), _ftm_eltype(Φ))}(undef, na_out * S, size(X, 2))
    mul!(Xout, Φ, X)
    return Xout
end


"""
    to_sparse(Φ::FactoredBasisMatrix)

Materialize a FactoredBasisMatrix into a full sparse matrix.
"""
function to_sparse(Φ::FactoredTransitionMatrix)
    S = length(Φ.Φs)
    na_in = size(Φ.Φs[1], 2)
    na_out = size(Φ.Φs[1], 1)
    N_out = na_out * S
    N_in = na_in * S
    if Φ.Π isa UniformScaling
        result = spzeros(_ftm_eltype(Φ), N_out, N_in)
        for s in 1:S
            result[(s-1)*na_out+1:s*na_out, (s-1)*na_in+1:s*na_in] = Φ.Φs[s]
        end
        return result
    else
        result = spzeros(_ftm_eltype(Φ), N_out, N_in)
        Π = Φ.Π
        for s_col in 1:S
            for s_row in 1:S
                if abs(Π[s_row, s_col]) > 1e-16
                    result[(s_row-1)*na_out+1:s_row*na_out,
                           (s_col-1)*na_in+1:s_col*na_in] = Π[s_row, s_col] * Φ.Φs[s_col]
                end
            end
        end
        return result
    end
end


# ============================================================
# PCHIP interpolation matrix constructors
# ============================================================

"""
    pchip_slope_matrix(x)

Compute the N×N sparse matrix D such that d = D * y gives the PCHIP slopes
at each grid point x[1] < x[2] < ... < x[N].

For cubic Hermite interpolation to reproduce cubics exactly, the slopes must
be exact for cubic polynomials. We use the **Bessel (parabolic)** formula at
interior points and **shape-preserving endpoint** formulas:

**Interior (k = 2, …, N−1) — Bessel formula:**
    d[k] = (h_k * s_{k-1} + h_{k-1} * s_k) / (h_{k-1} + h_k)

where h_{k-1} = x[k]−x[k−1], h_k = x[k+1]−x[k], s_{k-1} = (y[k]−y[k−1])/h_{k-1},
s_k = (y[k+1]−y[k])/h_k. This is the derivative of the parabola through
the three points (x[k−1], y[k−1]), (x[k], y[k]), (x[k+1], y[k+1]), evaluated
at x[k]. It is exact for polynomials up to degree 2.

**Endpoints — non-centered three-point formula:**
    d[1] = ((2h₁+h₂)*s₁ - h₁*s₂) / (h₁+h₂)
    d[N] = ((2hₙ₋₁+hₙ₋₂)*sₙ₋₁ - hₙ₋₁*sₙ₋₂) / (hₙ₋₁+hₙ₋₂)

These are also exact for polynomials up to degree 2.

**Note on cubic exactness:** The Bessel formula is exact for quadratics but
not for cubics. Cubic Hermite interpolation with Bessel slopes will therefore
reproduce quadratics exactly but will have O(h²) slope errors for cubics.
However, the *interpolation* error is still O(h⁴) for smooth functions because
the cubic Hermite basis is fourth-order accurate in function values even when
the slopes have second-order errors. For exact cubic reproduction, one would
need slopes computed from the exact cubic (which requires global information).
"""
function pchip_slope_matrix(x::AbstractVector)
    N = length(x)
    @assert N >= 2 "Need at least 2 points for PCHIP"
    Tv = float(eltype(x))          # weights inherit the grid's precision

    rows = Int[]
    cols = Int[]
    vals = Tv[]

    if N == 2
        h = x[2] - x[1]
        push!(rows, 1); push!(cols, 1); push!(vals, -1/h)
        push!(rows, 1); push!(cols, 2); push!(vals,  1/h)
        push!(rows, 2); push!(cols, 1); push!(vals, -1/h)
        push!(rows, 2); push!(cols, 2); push!(vals,  1/h)
        return sparse(rows, cols, vals, N, N)
    end

    # Left endpoint: one-sided three-point formula
    # d[1] = ((2h₁+h₂)*s₁ - h₁*s₂) / (h₁+h₂)
    h1 = x[2] - x[1]
    h2 = x[3] - x[2]
    # Expand in terms of y[1], y[2], y[3]:
    #   s₁ = (y[2]-y[1])/h₁,  s₂ = (y[3]-y[2])/h₂
    #   d[1] = -(2h₁+h₂)/(h₁(h₁+h₂)) y[1]
    #        + (2h₁+h₂)/(h₁(h₁+h₂)) y[2] - h₁/(h₂(h₁+h₂)) y[2]
    #        + h₁/(h₂(h₁+h₂)) y[3]        ... wait, sign:
    #   d[1] = (2h₁+h₂)/(h₁+h₂) * (y[2]-y[1])/h₁ - h₁/(h₁+h₂) * (y[3]-y[2])/h₂
    #        = -(2h₁+h₂)/(h₁(h₁+h₂)) * y[1]
    #        + [(2h₁+h₂)/(h₁(h₁+h₂)) + h₁/(h₂(h₁+h₂))] * y[2]
    #        - h₁/(h₂(h₁+h₂)) * y[3]
    c1_1 = -(2*h1+h2)/(h1*(h1+h2))
    c1_2 =  (2*h1+h2)/(h1*(h1+h2)) + h1/(h2*(h1+h2))
    c1_3 = -h1/(h2*(h1+h2))
    push!(rows, 1); push!(cols, 1); push!(vals, c1_1)
    push!(rows, 1); push!(cols, 2); push!(vals, c1_2)
    push!(rows, 1); push!(cols, 3); push!(vals, c1_3)

    # Interior points: Bessel (parabolic) formula
    # d[k] = (h_k * s_{k-1} + h_{k-1} * s_k) / (h_{k-1} + h_k)
    # where h_{k-1} = x[k]-x[k-1], h_k = x[k+1]-x[k]
    # In terms of y values:
    #   d[k] = -h_k/((h_{k-1}+h_k)*h_{k-1}) * y[k-1]
    #        + (h_k/((h_{k-1}+h_k)*h_{k-1}) - h_{k-1}/((h_{k-1}+h_k)*h_k)) * y[k]
    #        + h_{k-1}/((h_{k-1}+h_k)*h_k) * y[k+1]
    for k in 2:N-1
        hm = x[k] - x[k-1]    # h_{k-1}
        hp = x[k+1] - x[k]    # h_k
        wsum = hm + hp

        ck_m1 = -hp / (wsum * hm)
        ck_0  =  hp/(wsum*hm) - hm/(wsum*hp)
        ck_p1 =  hm / (wsum * hp)

        push!(rows, k); push!(cols, k-1); push!(vals, ck_m1)
        push!(rows, k); push!(cols, k);   push!(vals, ck_0)
        push!(rows, k); push!(cols, k+1); push!(vals, ck_p1)
    end

    # Right endpoint: one-sided three-point formula (mirror of left)
    # d[N] = ((2hₙ₋₁+hₙ₋₂)*sₙ₋₁ - hₙ₋₁*sₙ₋₂) / (hₙ₋₁+hₙ₋₂)
    h1 = x[N] - x[N-1]      # hₙ₋₁
    h2 = x[N-1] - x[N-2]    # hₙ₋₂
    cN_Nm2 =  h1/(h2*(h1+h2))
    cN_Nm1 = -(2*h1+h2)/(h1*(h1+h2)) - h1/(h2*(h1+h2))
    cN_N   =  (2*h1+h2)/(h1*(h1+h2))

    push!(rows, N); push!(cols, N-2); push!(vals, cN_Nm2)
    push!(rows, N); push!(cols, N-1); push!(vals, cN_Nm1)
    push!(rows, N); push!(cols, N);   push!(vals, cN_N)

    return sparse(rows, cols, vals, N, N)
end


"""
    pchip_eval_matrix(x_from, x_to)

Construct a sparse matrix P of size (length(x_to) × length(x_from)) such that
for any vector of values v at points x_from, P * v gives PCHIP-interpolated
values at points x_to.

Uses cubic Hermite basis functions:
    p(t) = h₀₀(t)*y[k] + h₁₀(t)*h*d[k] + h₀₁(t)*y[k+1] + h₁₁(t)*h*d[k+1]
where t = (x − x[k]) / h, h = x[k+1] − x[k], and
    h₀₀(t) = 2t³ − 3t² + 1
    h₁₀(t) = t³ − 2t² + t
    h₀₁(t) = −2t³ + 3t²
    h₁₁(t) = t³ − t²

Since slopes d[k] are linear functions of the data values (via pchip_slope_matrix),
the full evaluation is a linear function of the data values.
"""
function pchip_eval_matrix(x_from::AbstractVector, x_to::AbstractVector)
    N_from = length(x_from)
    N_to = length(x_to)
    Tv = float(promote_type(eltype(x_from), eltype(x_to)))

    D_slope = pchip_slope_matrix(x_from)

    rows = Int[]
    cols = Int[]
    vals = Tv[]

    for i in 1:N_to
        xt = x_to[i]

        # Find interval: k such that x_from[k] <= xt < x_from[k+1]
        k = searchsortedlast(x_from, xt)
        if k >= N_from
            k = N_from - 1
        elseif k < 1
            k = 1
        end

        h = x_from[k+1] - x_from[k]
        t = (xt - x_from[k]) / h

        # Hermite basis values
        h00 = 2*t^3 - 3*t^2 + 1
        h10 = t^3 - 2*t^2 + t
        h01 = -2*t^3 + 3*t^2
        h11 = t^3 - t^2

        # p = h00*y[k] + h01*y[k+1] + h*h10*(D_slope[k,:]*y) + h*h11*(D_slope[k+1,:]*y)
        w = zeros(Tv, N_from)
        w[k]   += h00
        w[k+1] += h01

        dk_row = D_slope[k, :]
        dk1_row = D_slope[k+1, :]
        for (j, v) in zip(findnz(dk_row)...)
            w[j] += h * h10 * v
        end
        for (j, v) in zip(findnz(dk1_row)...)
            w[j] += h * h11 * v
        end

        for j in 1:N_from
            if abs(w[j]) > 1e-16
                push!(rows, i)
                push!(cols, j)
                push!(vals, w[j])
            end
        end
    end

    return sparse(rows, cols, vals, N_to, N_from)
end


"""
    pchip_deriv_matrix(x_from, x_to)

Construct a sparse matrix P' of size (length(x_to) × length(x_from)) such that
for any vector of values v at points x_from, P' * v gives the derivative of the
PCHIP interpolant evaluated at points x_to.

The derivative of the cubic Hermite interpolant is:
    p'(x) = (1/h) * [h₀₀'(t)*y[k] + h₁₀'(t)*h*d[k] + h₀₁'(t)*y[k+1] + h₁₁'(t)*h*d[k+1]]
where t = (x − x[k])/h and:
    h₀₀'(t) = 6t² − 6t
    h₁₀'(t) = 3t² − 4t + 1
    h₀₁'(t) = −6t² + 6t
    h₁₁'(t) = 3t² − 2t
"""
function pchip_deriv_matrix(x_from::AbstractVector, x_to::AbstractVector)
    N_from = length(x_from)
    N_to = length(x_to)
    Tv = float(promote_type(eltype(x_from), eltype(x_to)))

    D_slope = pchip_slope_matrix(x_from)

    rows = Int[]
    cols = Int[]
    vals = Tv[]

    for i in 1:N_to
        xt = x_to[i]

        k = searchsortedlast(x_from, xt)
        if k >= N_from
            k = N_from - 1
        elseif k < 1
            k = 1
        end

        h = x_from[k+1] - x_from[k]
        t = (xt - x_from[k]) / h

        # Derivatives of Hermite basis w.r.t. x (divide by h for dt/dx = 1/h)
        dh00 = (6*t^2 - 6*t) / h
        dh10 = (3*t^2 - 4*t + 1)       # h cancels: h10'(t)/h * h = h10'(t)
        dh01 = (-6*t^2 + 6*t) / h
        dh11 = (3*t^2 - 2*t)           # same cancellation

        # p'(x) = dh00*y[k] + dh10*d[k] + dh01*y[k+1] + dh11*d[k+1]
        w = zeros(Tv, N_from)
        w[k]   += dh00
        w[k+1] += dh01

        dk_row = D_slope[k, :]
        dk1_row = D_slope[k+1, :]
        for (j, v) in zip(findnz(dk_row)...)
            w[j] += dh10 * v
        end
        for (j, v) in zip(findnz(dk1_row)...)
            w[j] += dh11 * v
        end

        for j in 1:N_from
            if abs(w[j]) > 1e-16
                push!(rows, i)
                push!(cols, j)
                push!(vals, w[j])
            end
        end
    end

    return sparse(rows, cols, vals, N_to, N_from)
end


# ============================================================
# Piecewise-LINEAR interpolation matrices + interp-mode switch
#
# Drop-in linear analogues of pchip_{eval,deriv}_matrix, same signature and
# sparse (N_to × N_from) layout.  On the interval x_from[k] ≤ xt < x_from[k+1]
# with h = x_from[k+1]-x_from[k], t = (xt-x_from[k])/h:
#   value      = (1-t)·y[k] + t·y[k+1]          (weights 1-t, t)
#   derivative = (y[k+1]-y[k])/h                (weights -1/h, 1/h; const on the cell)
# Boundary handling (k clamped to [1,N-1]) matches the PCHIP versions, so points
# outside the grid extrapolate off the edge interval identically.
#
# Motivation: PCHIP differentiates to a cubic whose slope can overshoot near a
# kink in the interpolated policy (the k′=0→interior portfolio splice), which can
# corrupt the household Jacobian in the perturbation.  Linear interp gives a
# bounded, piecewise-constant derivative that never overshoots.  Toggle globally
# with `INTERP_MODE[] = :linear` (default `:pchip`); every Φ builder below reads
# it through `_eval_matrix` / `_deriv_matrix`.
# ============================================================

const INTERP_MODE = Ref(:pchip)        # :pchip (default) or :linear

#  The EVALUATION (Φ) and DERIVATIVE (Φ_a) matrices can be built from
#  DIFFERENT interpolants.  The motivation above is specifically about the
#  derivative — PCHIP's cubic slope overshoots near the k′=0 portfolio splice
#  while linear's is bounded and piecewise constant — so the two roles are
#  worth separating rather than switching together.
#     INTERP_MODE_D[] = :follow  (default) — derivative follows INTERP_MODE
#     INTERP_MODE_D[] = :linear / :pchip   — derivative pinned independently
#  Default :follow reproduces the previous behaviour exactly.
const INTERP_MODE_D = Ref(:follow)
_deriv_mode() = INTERP_MODE_D[] === :follow ? INTERP_MODE[] : INTERP_MODE_D[]

function linear_eval_matrix(x_from::AbstractVector, x_to::AbstractVector)
    N_from = length(x_from); N_to = length(x_to)
    Tv = float(promote_type(eltype(x_from), eltype(x_to)))
    rows = Int[]; cols = Int[]; vals = Tv[]
    for i in 1:N_to
        k = clamp(searchsortedlast(x_from, x_to[i]), 1, N_from - 1)
        h = x_from[k+1] - x_from[k]
        t = (x_to[i] - x_from[k]) / h
        push!(rows, i); push!(cols, k);   push!(vals, 1 - t)
        push!(rows, i); push!(cols, k+1); push!(vals, t)
    end
    return sparse(rows, cols, vals, N_to, N_from)
end

function linear_deriv_matrix(x_from::AbstractVector, x_to::AbstractVector)
    N_from = length(x_from); N_to = length(x_to)
    Tv = float(promote_type(eltype(x_from), eltype(x_to)))
    rows = Int[]; cols = Int[]; vals = Tv[]
    for i in 1:N_to
        k = clamp(searchsortedlast(x_from, x_to[i]), 1, N_from - 1)
        h = x_from[k+1] - x_from[k]
        push!(rows, i); push!(cols, k);   push!(vals, -1/h)
        push!(rows, i); push!(cols, k+1); push!(vals,  1/h)
    end
    return sparse(rows, cols, vals, N_to, N_from)
end

# mode-dispatching 1-D builders — every tensor/Φ constructor calls THESE
_eval_matrix(x_from, x_to)  = INTERP_MODE[] === :linear ?
    linear_eval_matrix(x_from, x_to)  : pchip_eval_matrix(x_from, x_to)
_deriv_matrix(x_from, x_to) = _deriv_mode() === :linear ?
    linear_deriv_matrix(x_from, x_to) : pchip_deriv_matrix(x_from, x_to)


# ============================================================
# PCHIP integration (quadrature) constructors
# ============================================================

# Antiderivatives of the cubic Hermite basis functions on [0,1]:
#   Iαβ(t) = ∫₀ᵗ hαβ(s) ds, so ∫_a^b hαβ = Iαβ(b) − Iαβ(a).
#   h₀₀(t)=2t³−3t²+1,  h₁₀(t)=t³−2t²+t,  h₀₁(t)=−2t³+3t²,  h₁₁(t)=t³−t²
_pchip_I00(t) = t^4/2 - t^3 + t
_pchip_I10(t) = t^4/4 - 2*t^3/3 + t^2/2
_pchip_I01(t) = -t^4/2 + t^3
_pchip_I11(t) = t^4/4 - t^3/3


"""
    pchip_integral_weights(x)

Return the length-N quadrature weight vector `w` such that `dot(w, y)` equals the
exact integral of the PCHIP interpolant of `(x, y)` over the whole domain
`[x[1], x[N]]`.

Integrating the cubic Hermite form interval-by-interval gives, on `[x[k], x[k+1]]`
with `h = x[k+1]-x[k]`,
    ∫ p dx = (h/2)(y[k]+y[k+1]) + (h²/12)(d[k]-d[k+1]),
i.e. the composite trapezoid rule plus a slope correction. Summing over intervals
and using `d = D*y` (`D = pchip_slope_matrix(x)`) collapses the whole thing to a
single linear functional:
    w = w_trap + (1/12) * D' * c,
with `w_trap[j] = (h[j-1]+h[j])/2` (endpoints use only their one adjacent `h`) and
`c[j] = h[j]²·1[j≤N-1] − h[j-1]²·1[j≥2]` the coefficient on `d[j]`.

The result is *exact* for the interpolant (and hence exact for quadratics, which
the Bessel slopes reproduce). On a uniform grid the interior corrections telescope
and this reduces to the end-corrected trapezoid (Euler–Maclaurin) rule, O(h⁴).
"""
function pchip_integral_weights(x::AbstractVector)
    N = length(x)
    @assert N >= 2 "Need at least 2 points to integrate"
    Tv = float(eltype(x))
    h = diff(x)

    w_trap = zeros(Tv, N)
    c = zeros(Tv, N)                   # coefficient on d[j] in the slope correction
    for k in 1:N-1
        w_trap[k]   += h[k]/2
        w_trap[k+1] += h[k]/2
        c[k]   += h[k]^2
        c[k+1] -= h[k]^2
    end

    D = pchip_slope_matrix(x)
    return w_trap .+ (1/12) .* (D' * c)
end


"""
    pchip_integral_matrix(x_from, a, b)

Return the length-`length(x_from)` weight vector `w` such that `dot(w, y)` equals
the exact integral of the PCHIP interpolant of `(x_from, y)` over the sub-interval
`[a, b]`, where `a` and `b` need NOT lie on the grid.

Despite the name (kept for parallelism with `pchip_eval_matrix`/`pchip_deriv_matrix`)
this returns a single row functional — the definite-integral map is linear in `y`.
For `[a, b] = [x_from[1], x_from[end]]` it agrees with `pchip_integral_weights`.

On each grid interval `[x[k], x[k+1]]` overlapping `[a, b]`, with local coordinates
`α = (max(a,x[k])-x[k])/h`, `β = (min(b,x[k+1])-x[k])/h` (`h = x[k+1]-x[k]`), the
contribution is
    h·y[k]·ΔI00 + h²·d[k]·ΔI10 + h·y[k+1]·ΔI01 + h²·d[k+1]·ΔI11,
where `ΔIαβ = Iαβ(β) − Iαβ(α)` are the partial integrals of the Hermite bases.

The limits are clamped to `[x_from[1], x_from[end]]` (no extrapolation beyond the
grid). If `b < a` the sign is flipped, so `dot(w, y) = ∫_a^b`.
"""
function pchip_integral_matrix(x_from::AbstractVector, a::Real, b::Real)
    N = length(x_from)
    @assert N >= 2 "Need at least 2 points to integrate"
    Tv = float(eltype(x_from))

    D = pchip_slope_matrix(x_from)

    lo = clamp(min(a, b), x_from[1], x_from[N])
    hi = clamp(max(a, b), x_from[1], x_from[N])
    sgn = b < a ? -one(Tv) : one(Tv)

    w = zeros(Tv, N)
    for k in 1:N-1
        xl = x_from[k]
        xr = x_from[k+1]
        ol = max(lo, xl)          # overlap of [a,b] with this interval
        or = min(hi, xr)
        or <= ol && continue

        h = xr - xl
        α = (ol - xl) / h
        β = (or - xl) / h

        H00 = _pchip_I00(β) - _pchip_I00(α)
        H10 = _pchip_I10(β) - _pchip_I10(α)
        H01 = _pchip_I01(β) - _pchip_I01(α)
        H11 = _pchip_I11(β) - _pchip_I11(α)

        w[k]   += h * H00
        w[k+1] += h * H01

        dk_row  = D[k, :]
        dk1_row = D[k+1, :]
        for (j, v) in zip(findnz(dk_row)...)
            w[j] += h^2 * H10 * v
        end
        for (j, v) in zip(findnz(dk1_row)...)
            w[j] += h^2 * H11 * v
        end
    end

    return sgn .* w
end
