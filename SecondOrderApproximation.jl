#=============================================================================
    SecondOrderApproximation.jl   (BEM: z-variable formulation WITH DISCRETE
    CHOICE — the port of New Approximation Code/SecondOrderApproximationMPC.jl)

    Second-order approximation for the z-variable formulation, AD-ONLY: no
    second derivatives of F, G, f or pf are ever stored — every bilinear
    interaction is a nested-dual mixed directional derivative of the model
    functions themselves.  Pairs with FirstOrderApproximation.jl.

    ------------------------------------------------------------------------
    THEORY (per full node j; H = ∂²/∂α∂β F(z̄+αδ_t+βδ_k)|₀ the mixed partial
    of the FULL F along the two first-order directions, one dual call per
    node):

      y-block interaction:   η_tk = −Fyy⁻¹ H_y
      z-block interaction:   ztk  = f·(H_z + Fzy η_tk + Fze·ye′tk)
      full rows:             xtk  = [ztk;  y_z ztk + η_tk]
    with the same f as the first order and the genuinely-second-order part of
    the expectation argument
      ye′tk = E[ŷ′tk] + (∂_a E[ŷ′_k])·â_t + (∂_a E[ŷ′_t])·â_k + Eȳ_aa·(â_t, â_k),
    where (DC) the taste-integrated variations are
      ŷ_t   = fm x̂⁻_t + fp x̂⁺_t                                 (first order)
      ŷ_tk  = fm x⁻_tk + fp x⁺_tk + ∂²/∂α∂β f(x̄⁻+αx̂⁻_t+βx̂⁻_k, x̄⁺+…)  (Jensen term)
    from one nested-dual (f, pf) call per REDUCED node, which also gives the
    second-order choice-probability change
      p_tk  = pm x⁻_tk + pp x⁺_tk + ∂²/∂α∂β pf(…).

    Distribution (DC, pre-choice masses μ on the reduced fine grid; see the
    FO header), the mixed derivative of  μ_{t+1} = Σ_c Λc[c]( p_c ⊙ μ ) with
    displaced savings:
      μ̂tk_{t+1} = Λ̃ μ̂tk_t
        + (Λc[1] − Λc[2])( p̂t ⊙ μ̂k + p̂k ⊙ μ̂t ) + Mp·ptk
        + Σ_{c,d} D̄_d Λc[c]( ât_{d,c} ⊙ (p̄_c μ̂k + p̂k_c μ̄) + âk_{d,c} ⊙ (p̄_c μ̂t + p̂t_c μ̄) )
        + Σ_{c,d} M[d,c]·atk_{d,c}
        + Σ_{c,d,e} D̄²_{de} Λc[c]( ât_{d,c} ⊙ âk_{e,c} ⊙ ω̄_c )
        + kink terms (Mδ[d,c]·Ixtkδ),
    with p̂t_1 = p̂t, p̂t_2 = −p̂t (fine-grid interpolants).  Aggregation:
      ∫x dω̂tk = xtk·IntΦ + Wp ptk + Iμ μ̂tk
        + Σ_c x̂t_c Φ (p̄_c μ̂k) + Σ_c x̂k_c Φ (p̄_c μ̂t)               (policy × dist)
        + (x̂t_1 − x̂t_2)Φ (p̂k μ̄) + (x̂k_1 − x̂k_2)Φ (p̂t μ̄)          (policy × prob)
        + (x̄Φ_1 − x̄Φ_2)(p̂t ⊙ μ̂k + p̂k ⊙ μ̂t)                      (prob × dist),
    the first line in compute_XZZ!, the crosses in compute_Lemma2_ZZ_AD!.

    The σσ (aggregate-risk) block of the MPC code is NOT ported: the BEM
    transition experiment is deterministic (initial conditions X_0, μ̂_1 and
    no aggregate shocks), so only the (t,k) interaction chain is needed.

    Memory: only the t=1 slices of the (i,j) interactions are stored; the
    per-pair workspaces are reused; pair-invariant operators are cached (as
    in the MPC code).

    Kink corrections (ℵ) follow the MPC code (x̄Δ_b from FO.x̄_a, savings
    rows 1:n.a); their effect on the choice probability is not modelled (ℵ is
    empty in BEM).
=============================================================================#

include("FirstOrderApproximation.jl")


"""
SecondOrderApproximation (z-variable + discrete-choice variant; inherits the
precision Tf of its FirstOrderApproximation).
"""
@with_kw mutable struct SecondOrderApproximation{Tf<:AbstractFloat}
    FO::FirstOrderApproximation{Tf}
    T::Int = FO.T #Length of IRF

    #components of the derivative with respect to second state
    X_02::Vector{Tf} = zeros(Tf,1)
    Θ_02::Vector{Tf} = zeros(Tf,1)
    Ω̂k::Matrix{Tf} = zeros(Tf,1,1)
    ẑk::Array{Tf,3} = zeros(Tf,1,1,1)
    X̂k::Matrix{Tf} = zeros(Tf,1,1)

    ##Lemma 2 terms
    GΘΘtk::Matrix{Tf} = zeros(Tf,1,1) #nX×T

    ##Lemma 3 terms
    xtk::Array{Tf,3} = zeros(Tf,0,0,0)      # FULL rows [ztk; y_z ztk + η]
    ptk::Matrix{Tf} = zeros(Tf,0,0)         # (DC) second-order Prob(c=1) change, n.spr × T
    Ixtkδ::Array{Tf,3} = zeros(Tf,0,0,0)

    # dim-1 derivative operators for the kink correction (full / reduced grid)
    Φb::FactoredTransitionMatrix{UniformScaling{Bool},Tf}  = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=I)
    Φbr::FactoredTransitionMatrix{UniformScaling{Bool},Tf} = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=I)

    # ── reusable per-pair workspaces + pair-invariant caches ──
    ẑtk_ws::Array{Tf,3} = zeros(Tf,0,0,0)
    p̂t_ws::Matrix{Tf} = zeros(Tf,0,0)       # (DC) first-order prob paths of the pair
    p̂k_ws::Matrix{Tf} = zeros(Tf,0,0)
    DΛ_c::Matrix{FactoredTransitionMatrix{Matrix{Tf},Tf}} = Matrix{FactoredTransitionMatrix{Matrix{Tf},Tf}}(undef,0,0)
    D2Λ_c::Array{FactoredTransitionMatrix{Matrix{Tf},Tf},3} = Array{FactoredTransitionMatrix{Matrix{Tf},Tf}}(undef,0,0,0)
    Mδ_c::Matrix{FactoredTransitionMatrix{Matrix{Tf},Tf}} = Matrix{FactoredTransitionMatrix{Matrix{Tf},Tf}}(undef,0,0)
    ΛcΔ_c::FactoredTransitionMatrix{Matrix{Tf},Tf} = FactoredTransitionMatrix(Φs=[spzeros(Tf,1,1)], Π=ones(Tf,1,1))
    IntΦb_c::Vector{Tf} = zeros(Tf,0)
    x̄ΦI_c::Matrix{Tf} = zeros(Tf,0,0)       # Iμ[iI,:]  (n.I × n.Ω)
    x̄ΦIΔ_c::Matrix{Tf} = zeros(Tf,0,0)      # (x̄Φ₁ − x̄Φ₂)[iI,:]  (n.I × n.Ω)
    ρpow_c::Vector{Matrix{Tf}} = Matrix{Tf}[]

    #Outputs
    X̂tk::Matrix{Tf} = zeros(Tf,1,1)
    x̂tk::Array{Tf,3} = zeros(Tf,0,0,0)      # full path (full_x only)
    x̂tk1::Matrix{Tf} = zeros(Tf,0,0)        # t=1 slice (always)
    Ω̂tk::Matrix{Tf} = zeros(Tf,1,1)

    X̂_ΘΘ::Matrix{Matrix{Tf}} = Matrix{Matrix{Tf}}(undef,1,1)
    x̂_ΘΘ1::Matrix{Matrix{Tf}} = Matrix{Matrix{Tf}}(undef,1,1)
    Ixtkδ_ΘΘ1::Matrix{Matrix{Tf}} = Matrix{Matrix{Tf}}(undef,1,1)
    x̂_ΘΘ::Matrix{Array{Tf,3}} = Matrix{Array{Tf,3}}(undef,1,1)
    Ω̂_ΘΘ::Matrix{Matrix{Tf}} = Matrix{Matrix{Tf}}(undef,1,1)
end

#  keyword form `SecondOrderApproximation(FO=FO, ...)` — the precision comes
#  from FO, so the old un-parameterized call keeps working
function SecondOrderApproximation(; FO::FirstOrderApproximation, kwargs...)
    return SecondOrderApproximation{_fo_precision(FO)}(; FO=FO, kwargs...)
end
_fo_precision(::FirstOrderApproximation{Tf}) where {Tf} = Tf

function Base.copy(SO::SecondOrderApproximation)
    SOtemp = SecondOrderApproximation(FO=SO.FO)
    for k in fieldnames(SecondOrderApproximation)
        setfield!(SOtemp,k,getfield(SO,k))
    end
    return SOtemp
end


"""
    construct_Φb(ZO)

b-direction (dim-1) derivative operators, sparse nodes → fine grid, same
orientation/type as `ZO.Φ`: on the FULL policy grid (n.c·n.θ blocks) and on
the reduced grid (n.θ blocks).
"""
function construct_Φb(ZO::ZerothOrderApproximation{Tf}) where {Tf}
    @unpack agrids_sp, agrids_Ω, n = ZO
    tol = 1e-14
    blk = tensor_kron([d == 1 ? droptol!(_deriv_matrix(agrids_sp[d], agrids_Ω[d]), tol) :
                                droptol!(_eval_matrix(agrids_sp[d], agrids_Ω[d]), tol)
                       for d in 1:n.a])
    blk = _to_precision(Tf, sparse(blk'))       # RHS orientation, like Φ
    Φb  = FactoredTransitionMatrix(Φs=[copy(blk) for _ in 1:n.c*n.θ], Π=I)
    Φbr = FactoredTransitionMatrix(Φs=[copy(blk) for _ in 1:n.θ], Π=I)
    return Φb, Φbr
end


# ── pair-invariant cache accessors (built once per SO by the identical
#    expressions the call sites use)
function _so_IntΦb!(SO::SecondOrderApproximation)
    isempty(SO.IntΦb_c) && (SO.IntΦb_c = SO.Φb * SO.FO.ZO.ω̄)
    return SO.IntΦb_c
end

#  Iμ restricted to the Ilab rows — the observation matrix against μ̂
function _so_x̄ΦI!(SO::SecondOrderApproximation)
    size(SO.x̄ΦI_c, 1) == 0 && (SO.x̄ΦI_c = SO.FO.ZO.Iμ[SO.FO.ZO.iI, :])
    return SO.x̄ΦI_c
end

#  (x̄Φ₁ − x̄Φ₂) on the Ilab rows — the observation matrix of prob × dist crosses
function _so_x̄ΦIΔ!(SO::SecondOrderApproximation{Tf}) where {Tf}
    if size(SO.x̄ΦIΔ_c, 1) == 0
        ZO = SO.FO.ZO
        n = ZO.n
        x̄ΦI = reshape(ZO.x̄[ZO.iI, :] * ZO.Φ, length(ZO.iI), n.Ω, n.c)
        SO.x̄ΦIΔ_c = x̄ΦI[:,:,1] .- x̄ΦI[:,:,2]
    end
    return SO.x̄ΦIΔ_c
end

function _so_ρpow!(SO::SecondOrderApproximation)
    length(SO.ρpow_c) == SO.FO.T ||
        (SO.ρpow_c = [SO.FO.ZO.ρ_Θ^(t-1) for t in 1:SO.FO.T])
    return SO.ρpow_c
end

#  lifted operators per occupation (pair-invariant)
function _so_lifts!(SO::SecondOrderApproximation{Tf}) where {Tf}
    ZO = SO.FO.ZO
    @unpack n, Λc, D, D2 = ZO
    if size(SO.DΛ_c, 1) == 0
        SO.DΛ_c  = [D[d]*Λc[c] for d in 1:n.a, c in 1:n.c]
        SO.D2Λ_c = [D2[d,e]*Λc[c] for d in 1:n.a, e in 1:n.a, c in 1:n.c]
        Φs = [Λc[1].Φs[s] .- Λc[2].Φs[s] for s in 1:n.θ]
        SO.ΛcΔ_c = FactoredTransitionMatrix(Φs=Φs, Π=Matrix{Tf}(Λc[1].Π))
    end
    return SO.DΛ_c, SO.D2Λ_c, SO.ΛcΔ_c
end

function _so_Mδ!(SO::SecondOrderApproximation{Tf}) where {Tf}
    ZO = SO.FO.ZO
    @unpack n, Λc, D, ω̄ = ZO
    if size(SO.Mδ_c, 1) == 0
        Φbrt = transpose(SO.Φbr)
        SO.Mδ_c = Matrix{FactoredTransitionMatrix{Matrix{Tf},Tf}}(undef, n.a, n.c)
        for c in 1:n.c
            ω̄_c = diag_transition_matrix(ω̄[(c-1)*n.Ω+1:c*n.Ω], n.θ)
            for d in 1:n.a
                SO.Mδ_c[d,c] = D[d] * Λc[c] * ω̄_c * Φbrt
            end
        end
    end
    return SO.Mδ_c
end


"""
    _prob_path!(p̂, FO, ẑ, X̂)

(DC) First-order choice-probability path p̂[:,t] = pm·x̂e_t(·,1) + pp·x̂e_t(·,2)
from a tracked path ẑ (n.z × n.sp × T) and aggregates X̂ (n.X × T).
"""
function _prob_path!(p̂::AbstractMatrix, FO::FirstOrderApproximation{Tf},
                     ẑ::AbstractArray{Tf,3}, X̂::AbstractMatrix) where {Tf}
    ZO = FO.ZO
    n = ZO.n
    T = size(ẑ, 3)
    QX̂ = ZO.Q*X̂
    xe = zeros(Tf, n.e, n.sp)
    for t in 1:T
        reconstruct_rows!(xe, FO.Sez, FO.SeX, view(ẑ,:,:,t), view(QX̂,:,t))
        choice_prob!(view(p̂,:,t), xe, ZO)
    end
    return p̂
end

#  the pair's two first-order probability paths (reused by every ZZ routine)
function _so_prob_paths!(SO::SecondOrderApproximation{Tf}) where {Tf}
    FO = SO.FO
    n = FO.ZO.n
    Tt = size(FO.ẑt, 3)
    Tk = size(SO.ẑk, 3)
    size(SO.p̂t_ws) == (n.spr, Tt) || (SO.p̂t_ws = zeros(Tf, n.spr, Tt))
    size(SO.p̂k_ws) == (n.spr, Tk) || (SO.p̂k_ws = zeros(Tf, n.spr, Tk))
    _prob_path!(SO.p̂t_ws, FO, FO.ẑt, FO.X̂t)
    _prob_path!(SO.p̂k_ws, FO, SO.ẑk, SO.X̂k)
    return SO.p̂t_ws, SO.p̂k_ws
end


"""
    compute_Lemma2_ZZ_AD!(SO, inputs)

All-first-order interactions of G as the mixed second directional derivative
∂²/∂α∂β of G along the two first-order paths (nested ForwardDiff), plus the
first-order × first-order crosses inside the integrated policy
(policy × distribution, policy × probability, probability × distribution —
see the file header), applied through dG.x.
"""
function compute_Lemma2_ZZ_AD!(SO::SecondOrderApproximation{Tf}, inputs::Inputs) where {Tf}
    @unpack FO, Ω̂k, ẑk, X̂k, Θ_02 = SO
    @unpack Ω̂t, ẑt, X̂t, Θ_0, ZO = FO
    @unpack x̄, X̄, Φ, Φr, ω̄, μ̄, p̄, dG, ρ_Θ, P, Q, n, iI, IntΦ, Wp, Iμ = ZO
    @unpack G, Θ̄ = inputs
    T = size(ẑk)[end]
    nI = length(iI)

    size(SO.GΘΘtk) == (n.X, T) || (SO.GΘΘtk = zeros(Tf, n.X, T))
    GΘΘtk = SO.GΘΘtk                    # every column fully written below
    ρpow = _so_ρpow!(SO)
    p̂t_all, p̂k_all = _so_prob_paths!(SO)
    x̄ΦIΔ = _so_x̄ΦIΔ!(SO)
    X̄_ = P*X̄
    Ix̄ = x̄ * IntΦ
    QX̂t_all = Q*X̂t
    QX̂k_all = Q*X̂k
    dGx_I = dG.x[:, iI]
    Φrt = transpose(Φr)
    x̂t_t = zeros(Tf, n.x, n.sp)
    x̂k_t = zeros(Tf, n.x, n.sp)
    #  hoisted per-t buffers
    b1 = zeros(Tf, n.x);       b2 = zeros(Tf, n.x);     b3 = zeros(Tf, n.x)
    x̂It = zeros(Tf, n.x);      x̂Ik = zeros(Tf, n.x)
    bP1 = zeros(Tf, n.A);      bP2 = zeros(Tf, n.A)
    θt = zeros(Tf, n.Θ);       θk = zeros(Tf, n.Θ)
    bX = zeros(Tf, n.X)
    zX = zeros(Tf, n.X)                 # the t = T "no expectation" argument
    x̂tI = zeros(Tf, nI, n.sp);  x̂kI = zeros(Tf, nI, n.sp)
    XtF = zeros(Tf, nI, n.Ω, n.c);  XkF = zeros(Tf, nI, n.Ω, n.c)
    p̂tf = zeros(Tf, n.Ω);  p̂kf = zeros(Tf, n.Ω)
    wv = zeros(Tf, n.Ω)
    cross = zeros(Tf, nI);  bI = zeros(Tf, nI)
    XΔ = zeros(Tf, nI, n.Ω)
    for t in 1:T
        reconstruct_rows!(x̂t_t, FO.Wz, FO.WX, view(ẑt,:,:,t), view(QX̂t_all,:,t))
        reconstruct_rows!(x̂k_t, FO.Wz, FO.WX, view(ẑk,:,:,t), view(QX̂k_all,:,t))
        X̂t_t = @view X̂t[:,t]
        X̂k_t = @view X̂k[:,t]
        p̂t_t = @view p̂t_all[:,t]
        p̂k_t = @view p̂k_all[:,t]
        # first-order integrals  Ix̂ = x̂·IntΦ + Wp p̂ + Iμ μ̂
        mul!(b1, x̂t_t, IntΦ);  mul!(b2, Wp, p̂t_t);  mul!(b3, Iμ, view(Ω̂t,:,t));  x̂It .= b1 .+ b2 .+ b3
        mul!(b1, x̂k_t, IntΦ);  mul!(b2, Wp, p̂k_t);  mul!(b3, Iμ, view(Ω̂k,:,t));  x̂Ik .= b1 .+ b2 .+ b3
        if t == 1
            X̂t_t_ = FO.X_0
            X̂k_t_ = SO.X_02
        else
            X̂t_t_ = mul!(bP1, P, view(X̂t,:,t-1))
            X̂k_t_ = mul!(bP2, P, view(X̂k,:,t-1))
        end
        if t == T
            X̂t_tᵉ = zX
            X̂k_tᵉ = zX
        else
            X̂t_tᵉ = @view X̂t[:,t+1]
            X̂k_tᵉ = @view X̂k[:,t+1]
        end

        mul!(θt, ρpow[t], Θ_0)
        mul!(θk, ρpow[t], Θ_02)

        Gtk = (α,β) -> G(Ix̄ .+ α.*x̂It  .+ β.*x̂Ik,
                         X̄_ .+ α.*X̂t_t_ .+ β.*X̂k_t_,
                         X̄  .+ α.*X̂t_t  .+ β.*X̂k_t,
                         X̄  .+ α.*X̂t_tᵉ .+ β.*X̂k_tᵉ,
                         Θ̄  .+ α.*θt    .+ β.*θk)
        GΘΘtk[:,t] = ForwardDiff.derivative(β->ForwardDiff.derivative(α->Gtk(α,β),Tf(0.0)),Tf(0.0))

        # second-order crosses inside the integrated policy (Ilab rows)
        x̂tI .= @view x̂t_t[iI, :]
        x̂kI .= @view x̂k_t[iI, :]
        mul!(XtF, x̂tI, Φ)                       # (nI, n.Ω, n.c) fine-grid values
        mul!(XkF, x̂kI, Φ)
        mul!(p̂tf, Φrt, p̂t_t)
        mul!(p̂kf, Φrt, p̂k_t)
        Ω̂t_t = @view Ω̂t[:,t]
        Ω̂k_t = @view Ω̂k[:,t]
        # policy × distribution:  Σ_c x̂t_c Φ (p̄_c ⊙ μ̂k) + (t↔k)
        wv .= p̄ .* Ω̂k_t;                 @views mul!(cross, XtF[:,:,1], wv)
        wv .= (one(Tf) .- p̄) .* Ω̂k_t;    @views mul!(cross, XtF[:,:,2], wv, one(Tf), one(Tf))
        wv .= p̄ .* Ω̂t_t;                 @views mul!(cross, XkF[:,:,1], wv, one(Tf), one(Tf))
        wv .= (one(Tf) .- p̄) .* Ω̂t_t;    @views mul!(cross, XkF[:,:,2], wv, one(Tf), one(Tf))
        # policy × probability:  (x̂t_1 − x̂t_2)Φ (p̂k ⊙ μ̄) + (t↔k)
        @views XΔ .= XtF[:,:,1] .- XtF[:,:,2]
        wv .= p̂kf .* μ̄;                   mul!(cross, XΔ, wv, one(Tf), one(Tf))
        @views XΔ .= XkF[:,:,1] .- XkF[:,:,2]
        wv .= p̂tf .* μ̄;                   mul!(cross, XΔ, wv, one(Tf), one(Tf))
        # probability × distribution:  (x̄Φ_1 − x̄Φ_2)(p̂t ⊙ μ̂k + p̂k ⊙ μ̂t)
        wv .= p̂tf .* Ω̂k_t .+ p̂kf .* Ω̂t_t; mul!(cross, x̄ΦIΔ, wv, one(Tf), one(Tf))
        mul!(bX, dGx_I, cross)
        GΘΘtk[:,t] .+= bX
    end
end


# ──────────────────────────────────────────────────────────────────────────
# Nested-dual machinery for the in-place mixed second directional derivative
# (identical to the MPC file).
# ──────────────────────────────────────────────────────────────────────────
struct MixedTag1 end
struct MixedTag2 end

@inline mixed_make_dual(z̄i, δti, δki) =
    ForwardDiff.Dual{MixedTag1}(ForwardDiff.Dual{MixedTag2}(z̄i, δki),
                                ForwardDiff.Dual{MixedTag2}(δti, zero(z̄i)))

@inline mixed_partial(d) = ForwardDiff.partials(ForwardDiff.partials(d, 1), 1)

const MixedD2{Tf} = ForwardDiff.Dual{MixedTag2,Tf,1}
const MixedD1{Tf} = ForwardDiff.Dual{MixedTag1,MixedD2{Tf},1}

# Function barriers so the non-concrete inputs.F / f / pf are specialized on
@inline function eval_mixed_F!(dFtemp, F::FT, j, θ, a_, c, xd, Xd, ye′d) where {FT}
    yd = F(j, θ, a_, c, xd, Xd, ye′d)
    @inbounds for o in eachindex(dFtemp)
        dFtemp[o] = mixed_partial(yd[o])
    end
    return dFtemp
end

@inline function eval_mixed_fp!(d2f, f::FT, pf::PT, xmd, xpd) where {FT,PT}
    yd = f(xmd, xpd)
    @inbounds for o in eachindex(d2f)
        d2f[o] = mixed_partial(yd[o])
    end
    return mixed_partial(pf(xmd, xpd))
end

"""
    mixed_fp!(D2f, D2p, x̄, x̂et, x̂ek, ZO, f, pf, bufs)

(DC) Jensen terms of the discrete choice at every REDUCED node: the mixed
second directional derivatives of f (into D2f, n.ye × n.spr) and pf (into
D2p, n.spr) along the two first-order variations x̂et, x̂ek (n.e × n.sp, xelab
rows at both occupations).  `bufs` are per-chunk dual buffers (see
_mixed_fp_bufs).
"""
function mixed_fp!(D2f::AbstractMatrix, D2p::AbstractVector, x̄::AbstractMatrix,
                   x̂et::AbstractMatrix, x̂ek::AbstractMatrix,
                   ZO::ZerothOrderApproximation{Tf}, f::FT, pf::PT, bufs, nt::Int) where {Tf,FT,PT}
    n = ZO.n
    ie = ZO.ie
    threaded_foreach(n.spr, ntasks=nt) do chunk, jrs
        @unpack xmd, xpd, d2f = bufs[chunk]
        for jr in jrs
            jp = jr + n.spr
            @inbounds for i in 1:n.x
                xmd[i] = mixed_make_dual(x̄[i,jr], zero(Tf), zero(Tf))
                xpd[i] = mixed_make_dual(x̄[i,jp], zero(Tf), zero(Tf))
            end
            @inbounds for (r, i) in enumerate(ie)
                xmd[i] = mixed_make_dual(x̄[i,jr], x̂et[r,jr], x̂ek[r,jr])
                xpd[i] = mixed_make_dual(x̄[i,jp], x̂et[r,jp], x̂ek[r,jp])
            end
            D2p[jr] = eval_mixed_fp!(d2f, f, pf, xmd, xpd)
            @inbounds for v in 1:n.ye
                D2f[v,jr] = d2f[v]
            end
        end
    end
    return D2f
end

_mixed_fp_bufs(::Type{Tf}, n::Nums, nt::Int) where {Tf} =
    [(xmd = Vector{MixedD1{Tf}}(undef, n.x), xpd = Vector{MixedD1{Tf}}(undef, n.x),
      d2f = zeros(Tf, n.ye)) for _ in 1:nt]


"""
    compute_lemma3_ZZ_AD_inplace!(SO, inputs)

The (t,k) interaction of the individual block, z-form (see the file header):
backward in s, one nested-dual F evaluation per node giving H, then
    η_tk = −invFyy·H_y,     ztk = f·(H_z + Fzy η_tk + Fze·ye′tk),
with SO.xtk storing the reconstructed FULL rows [ztk; y_z ztk + η_tk] and
SO.ptk the second-order choice-probability change (DC).
"""
function compute_lemma3_ZZ_AD_inplace!(SO::SecondOrderApproximation{Tf}, inputs::Inputs) where {Tf}
    @unpack FO, X̂k, ẑk = SO
    @unpack ZO, ẑt, X̂t = FO
    @unpack x̄, ȳ, X̄, Φ̃ᵉₐ, Φ̃ᵉ, Q, n, dF, aθ_sp, ie = ZO
    @unpack F, f, pf = inputs
    T = size(ẑk)[end]
    nz, ny, ne, nye, na = n.z, n.y, n.e, n.ye, n.a
    zi, yi = 1:nz, nz+1:n.x

    # Steady-state arguments of F (the point the Hessian is evaluated at)
    Eȳ′ = ȳ * Φ̃ᵉ
    argX̄ = Q*X̄

    # SS expectation derivatives (first and second) of the taste-integrated
    # function: ȳ_a = fm x̄⁻_a + fp x̄⁺_a at the nodes (chain rule), then the
    # differentiate-splines-at-most-once convention for Eȳ_aa.  Kept
    # FLATTENED as (nye·na, na, n.sp) so the per-node contraction with
    # (ât, âk) is two plain gemvs.
    Eȳ_a = FO.Eȳ_a
    ȳ_a = zeros(Tf, nye, na, n.sp)
    taste_mul!(ȳ_a, FO.x̄_a[ie,:,:], ZO)
    Eȳ_aa = reshape(reshape(ȳ_a, nye*na, n.sp)*Φ̃ᵉₐ, nye*na, na, n.sp)

    #Preallocate arrays (s-level arrays SHARED, read-only inside the j loop;
    #per-node scratch replicated once per chunk)
    Eŷt_a_s = zeros(Tf,nye,na,n.sp)
    Eŷk_a_s = zeros(Tf,nye,na,n.sp)
    Eŷt_s = zeros(Tf,nye,n.sp)
    Eŷk_s = zeros(Tf,nye,n.sp)
    Eytk = zeros(Tf,nye,n.sp)
    x̂t_s = zeros(Tf,n.x,n.sp)
    x̂k_s = zeros(Tf,n.x,n.sp)
    x̂et_next = zeros(Tf,ne,n.sp)
    x̂ek_next = zeros(Tf,ne,n.sp)
    ŷt_next = zeros(Tf,nye,n.sp)
    ŷk_next = zeros(Tf,nye,n.sp)
    xetk_next = zeros(Tf,ne,n.sp)
    ytk_next = zeros(Tf,nye,n.sp)
    xetk_cur = zeros(Tf,ne,n.sp)
    D2f_cur = zeros(Tf,nye,n.spr);  D2f_next = zeros(Tf,nye,n.spr)
    D2p_cur = zeros(Tf,n.spr)
    size(SO.xtk) == (n.x, n.sp, T) || (SO.xtk = Array{Tf,3}(undef, n.x, n.sp, T))
    size(SO.ptk) == (n.spr, T)     || (SO.ptk = Array{Tf,2}(undef, n.spr, T))
    xtk = SO.xtk
    ptk = SO.ptk
    QX̂t_all = Q*X̂t
    QX̂k_all = Q*X̂k

    Xd  = Vector{MixedD1{Tf}}(undef, n.Q)

    nθc = size(aθ_sp, 2) - na
    nt = n_tasks(n.sp)
    bufs = [begin
             c1 = zeros(Tf, nye*na)
             (dFtemp = zeros(Tf, n.x),
              δye′_t = zeros(Tf, nye),
              δye′_k = zeros(Tf, nye),
              ye′tk  = zeros(Tf, nye),
              ηbuf   = zeros(Tf, ny),
              tmpz   = zeros(Tf, nz),
              ztkbuf = zeros(Tf, nz),
              a_buf  = zeros(Tf, na),
              θ_buf  = zeros(Tf, nθc),
              c1     = c1,
              c1m    = reshape(c1, nye, na),
              c2     = zeros(Tf, nye),
              xd     = Vector{MixedD1{Tf}}(undef, n.x),
              ye′d   = Vector{MixedD1{Tf}}(undef, nye))
            end for _ in 1:nt]
    ntr = n_tasks(n.spr)
    fpbufs = _mixed_fp_bufs(Tf, n, ntr)

    for s in reverse(1:T)
        # current-period full first-order reconstructions
        reconstruct_rows!(x̂t_s, FO.Wz, FO.WX, view(ẑt,:,:,s), view(QX̂t_all,:,s))
        reconstruct_rows!(x̂k_s, FO.Wz, FO.WX, view(ẑk,:,:,s), view(QX̂k_all,:,s))
        # (DC) Jensen terms of f and pf at period s
        mixed_fp!(D2f_cur, D2p_cur, x̄, view(x̂t_s, ie, :), view(x̂k_s, ie, :), ZO, f, pf, fpbufs, ntr)

        if s < T
            reconstruct_rows!(x̂et_next, FO.Sez, FO.SeX, view(ẑt,:,:,s+1), view(QX̂t_all,:,s+1))
            reconstruct_rows!(x̂ek_next, FO.Sez, FO.SeX, view(ẑk,:,:,s+1), view(QX̂k_all,:,s+1))
            taste_mul!(ŷt_next, x̂et_next, ZO)
            taste_mul!(ŷk_next, x̂ek_next, ZO)
            mul!(Eŷt_a_s, ŷt_next, Φ̃ᵉₐ)
            mul!(Eŷk_a_s, ŷk_next, Φ̃ᵉₐ)
            mul!(Eŷt_s, ŷt_next, Φ̃ᵉ)
            mul!(Eŷk_s, ŷk_next, Φ̃ᵉ)
            threaded_foreach(n.sp) do _, jr
                @inbounds for j in jr, r in 1:ne
                    xetk_next[r,j] = xtk[ie[r],j,s+1]
                end
            end
            taste_mul!(ytk_next, xetk_next, ZO)
            @inbounds for jr in 1:n.spr, v in 1:nye
                ytk_next[v,jr] += D2f_next[v,jr]
                ytk_next[v,jr+n.spr] += D2f_next[v,jr]
            end
            mul!(Eytk, ytk_next, Φ̃ᵉ)
        else
            Eŷt_a_s .= zero(Tf)
            Eŷk_a_s .= zero(Tf)
            Eŷt_s .= zero(Tf)
            Eŷk_s .= zero(Tf)
            Eytk .= zero(Tf)
        end
        X̂t_s = @view QX̂t_all[:,s]
        X̂k_s = @view QX̂k_all[:,s]
        @inbounds for i in 1:n.Q
            Xd[i] = mixed_make_dual(argX̄[i], X̂t_s[i], X̂k_s[i])
        end

        threaded_foreach(n.sp, ntasks=nt) do chunk, jrange
            @unpack dFtemp, δye′_t, δye′_k, ye′tk, ηbuf, tmpz, ztkbuf,
                    a_buf, θ_buf, c1, c1m, c2, xd, ye′d = bufs[chunk]
            for j in jrange
                c  = (j-1) ÷ n.spr + 1
                jr = j - (c-1)*n.spr
                @inbounds for d in 1:na
                    a_buf[d] = aθ_sp[jr,d]
                end
                @inbounds for d in 1:nθc
                    θ_buf[d] = aθ_sp[jr,na+d]
                end
                a_ = a_buf
                θv = θ_buf
                argx̄   = @view x̄[:,j]
                argEye = @view Eȳ′[:,j]
                ât_sj = @view ẑt[1:na,j,s]           # savings = z-rows 1:n.a
                âk_sj = @view ẑk[1:na,j,s]

                # First-order perturbations of the F arguments in directions t and k
                δx_t  = @view x̂t_s[:,j]
                δx_k  = @view x̂k_s[:,j]
                δye′_t .= @view Eŷt_s[:,j]
                δye′_k .= @view Eŷk_s[:,j]
                @views mul!(δye′_t, Eȳ_a[:,:,j], ât_sj, one(Tf), one(Tf))
                @views mul!(δye′_k, Eȳ_a[:,:,j], âk_sj, one(Tf), one(Tf))

                @inbounds for i in 1:n.x
                    xd[i]  = mixed_make_dual(argx̄[i], δx_t[i], δx_k[i])
                end
                @inbounds for i in 1:nye
                    ye′d[i] = mixed_make_dual(argEye[i], δye′_t[i], δye′_k[i])
                end
                eval_mixed_F!(dFtemp, F, j, θv, a_, c, xd, Xd, ye′d)

                # Genuinely second-order part of the expectation argument
                ye′tk .= @view Eytk[:,j]
                @views mul!(ye′tk, Eŷk_a_s[:,:,j], ât_sj, one(Tf), one(Tf))
                @views mul!(ye′tk, Eŷt_a_s[:,:,j], âk_sj, one(Tf), one(Tf))
                @views mul!(c1, Eȳ_aa[:,:,j], âk_sj)
                mul!(c2, c1m, ât_sj)
                ye′tk .+= c2

                # η = −invFyy·H_y ;  ztk = f·(H_z + Fzy η + Fze ye′tk)
                @views mul!(ηbuf, FO.invFyy[:,:,j], dFtemp[yi])
                ηbuf .*= -1
                tmpz .= @view dFtemp[zi]
                @views mul!(tmpz, dF.x[j][zi,yi], ηbuf, one(Tf), one(Tf))
                @views mul!(tmpz, dF.xe[j][zi,:], ye′tk, one(Tf), one(Tf))
                @views mul!(ztkbuf, FO.f[:,:,j], tmpz)

                # store FULL rows: z-rows = ztk; y-rows = y_z ztk + η
                xtk[zi,j,s] .= ztkbuf
                ηy = @view xtk[yi,j,s]
                @views mul!(ηy, FO.y_z[:,:,j], ztkbuf)
                ηy .+= ηbuf
            end
        end

        # (DC) second-order probability change at s: pm x⁻tk + pp x⁺tk + Jensen
        threaded_foreach(n.sp) do _, jr
            @inbounds for j in jr, r in 1:ne
                xetk_cur[r,j] = xtk[ie[r],j,s]
            end
        end
        choice_prob!(view(ptk,:,s), xetk_cur, ZO)
        @views ptk[:,s] .+= D2p_cur
        D2f_next .= D2f_cur
    end
end


"""
    compute_lemma3_ZZ_kink!(SO)

Kink (moving mass-point) correction (as in the MPC file): x̄Δ_b from FO.x̄_a
(full rows), savings differences from the tracked ẑ (rows 1:n.a).  With ℵ
empty fills SO.Ixtkδ with zeros and returns.
"""
function compute_lemma3_ZZ_kink!(SO::SecondOrderApproximation{Tf}) where {Tf}
    @unpack FO, ẑk = SO
    @unpack ZO, ẑt, x̄_a = FO
    @unpack n, ℵ, agrids_sp = ZO
    nℵ = length(ℵ)
    T = size(ẑk)[end]
    na = n.a

    if size(SO.Ixtkδ) != (n.x, n.sp, T)
        SO.Ixtkδ = zeros(Tf, n.x, n.sp, T)
    else
        nℵ == 0 ? fill!(view(SO.Ixtkδ,:,:,1), zero(Tf)) : fill!(SO.Ixtkδ, zero(Tf))
    end
    Ixtkδ = SO.Ixtkδ
    nℵ == 0 && return

    x̄Δ_b = x̄_a[:,1,ℵ.+1] .- x̄_a[:,1,ℵ]         # n.x × nℵ
    āΔ_b = x̄Δ_b[1:na,:]                          # savings rows (z-rows 1:n.a)
    nrm2 = vec(sum(abs2, āΔ_b, dims=1))
    inv_nrm2 = [g > 1e-10 ? one(Tf)/g : zero(Tf) for g in nrm2]
    xtkδ = zeros(Tf,n.x,nℵ,T)
    for t in 1:T
        âΔ_t = ẑt[1:na,ℵ.+1,t] .- ẑt[1:na,ℵ,t]
        âΔ_k = ẑk[1:na,ℵ.+1,t] .- ẑk[1:na,ℵ,t]
        K̂_t = .-inv_nrm2 .* vec(sum(āΔ_b .* âΔ_t, dims=1))
        K̂_k = .-inv_nrm2 .* vec(sum(āΔ_b .* âΔ_k, dims=1))
        xtkδ[:,:,t] .= x̄Δ_b .* reshape(K̂_t,1,:) .* reshape(K̂_k,1,:)
    end
    # step function: the tail of the cutoff's b-line (dim-1-fastest grid)
    Nb1 = length(agrids_sp[1])
    for (iℵ,j) in enumerate(ℵ)
        mask_stepfunction = (j+1):(cld(j,Nb1)*Nb1)
        Ixtkδ[:,mask_stepfunction,:] .+= reshape(xtkδ[:,iℵ,:],n.x,1,T)
    end
end


"""
    compute_Corollary2_ZZ!(SO)

Second-order law of motion of the PRE-CHOICE distribution (file header):
plain transition, probability × distribution crosses, the second-order
probability change, per-occupation D̄ crosses (savings × distribution and
savings × probability), the lifted second-order savings, the per-occupation
D̄² curvature term, and the kink term.
"""
function compute_Corollary2_ZZ!(SO::SecondOrderApproximation{Tf}) where {Tf}
    @unpack FO, xtk, ptk, Ixtkδ = SO
    @unpack ZO, M, Mp, Ω̂t, ẑt = FO
    @unpack n, Φr, ω̄, μ̄, p̄, Λ, ℵ = ZO
    @unpack Ω̂k, ẑk = SO
    T = size(ẑk)[end]
    na, nc = n.a, n.c

    kinks = !isempty(ℵ)
    DΛ, D2Λ, ΛcΔ = _so_lifts!(SO)
    Mδ = kinks ? _so_Mδ!(SO) : SO.Mδ_c
    p̂t_all, p̂k_all = _so_prob_paths!(SO)
    Φrt = transpose(Φr)

    if size(SO.Ω̂tk) != (n.Ω, T)
        SO.Ω̂tk = zeros(Tf, n.Ω, T)
    else
        fill!(view(SO.Ω̂tk,:,1), zero(Tf))
    end
    Ω̂tk = SO.Ω̂tk
    âts = zeros(Tf, na, n.sp)              # contiguous savings buffers
    âks = zeros(Tf, na, n.sp)
    ât  = zeros(Tf, na, n.Ω, nc)           # fine-grid savings per occupation
    âk  = zeros(Tf, na, n.Ω, nc)
    p̂tf = zeros(Tf, n.Ω);  p̂kf = zeros(Tf, n.Ω)
    wt  = zeros(Tf, n.Ω);  wk = zeros(Tf, n.Ω)
    atk_d  = zeros(Tf, n.spr)
    Iatkδ_d = zeros(Tf, n.spr)
    bc   = zeros(Tf, n.Ω)                  # broadcast argument buffer
    tmpΩ = zeros(Tf, n.Ω)                  # operator-product buffer
    for t in 2:T
        âts .= @view ẑt[1:na,:,t-1]
        âks .= @view ẑk[1:na,:,t-1]
        for c in 1:nc
            blk = (c-1)*n.spr+1:c*n.spr
            @views mul!(ât[:,:,c], âts[:,blk], Φr)
            @views mul!(âk[:,:,c], âks[:,blk], Φr)
        end
        @views mul!(p̂tf, Φrt, p̂t_all[:,t-1])
        @views mul!(p̂kf, Φrt, p̂k_all[:,t-1])
        Ω̂t_ = @view Ω̂t[:,t-1]
        Ω̂k_ = @view Ω̂k[:,t-1]
        Ω̂tk_t = @view(Ω̂tk[:,t])
        mul!(Ω̂tk_t, Λ, @view(Ω̂tk[:,t-1]))
        # probability × distribution crosses and the second-order probability
        bc .= p̂tf .* Ω̂k_ .+ p̂kf .* Ω̂t_
        mul!(tmpΩ, ΛcΔ, bc)
        Ω̂tk_t .+= tmpΩ
        @views mul!(tmpΩ, Mp, ptk[:,t-1])
        Ω̂tk_t .+= tmpΩ
        for c in 1:nc
            sgn = c == 1 ? one(Tf) : -one(Tf)
            p̄c = c == 1 ? p̄ : (one(Tf) .- p̄)
            ω̄c = @view ω̄[(c-1)*n.Ω+1:c*n.Ω]
            wt .= p̄c .* Ω̂t_ .+ sgn .* p̂tf .* μ̄       # weight multiplying âk
            wk .= p̄c .* Ω̂k_ .+ sgn .* p̂kf .* μ̄       # weight multiplying ât
            blk = (c-1)*n.spr+1:c*n.spr
            for d in 1:na
                atk_d .= @view xtk[d,blk,t-1]
                mul!(tmpΩ, M[d,c], atk_d)
                Ω̂tk_t .+= tmpΩ
                bc .= @view(ât[d,:,c]).*wk .+ @view(âk[d,:,c]).*wt
                mul!(tmpΩ, DΛ[d,c], bc)
                Ω̂tk_t .+= tmpΩ
                for e in 1:na
                    bc .= @view(ât[d,:,c]).*@view(âk[e,:,c]).*ω̄c
                    mul!(tmpΩ, D2Λ[d,e,c], bc)
                    Ω̂tk_t .+= tmpΩ
                end
                if kinks
                    Iatkδ_d .= @view Ixtkδ[d,blk,t-1]
                    mul!(tmpΩ, Mδ[d,c], Iatkδ_d)
                    Ω̂tk_t .+= tmpΩ
                end
            end
        end
    end
end

"""
    compute_XZZ!(SO)

Computes the path of the second order derivatives X_ZZ (Ilab rows of the
integrals only — dG.x is zero elsewhere).
"""
function compute_XZZ!(SO::SecondOrderApproximation{Tf}) where {Tf}
    @unpack FO, ẑk, GΘΘtk, Ω̂tk, xtk, ptk, Ixtkδ = SO
    @unpack ZO, luBB = FO
    @unpack dG, n, ℵ, iI, IntΦ, Wp = ZO
    T = size(FO.X̂t,2)
    T2 = size(ẑk)[end]

    x̄ΦI = _so_x̄ΦI!(SO)            # n.I × n.Ω
    WpI = Wp[iI, :]
    dGx_I = dG.x[:, iI]
    kinks = !isempty(ℵ)
    IntΦb = kinks ? _so_IntΦb!(SO) : Tf[]

    AA = zeros(Tf, n.X,T)
    nt = n_tasks(T2, min_chunk=1)
    bufs = [(IBC = zeros(Tf, n.I), Intxtk = zeros(Tf, n.I), Ipk = zeros(Tf, n.I)) for _ in 1:nt]
    threaded_foreach(T2, ntasks=nt) do chunk, tr
        @unpack IBC, Intxtk, Ipk = bufs[chunk]
        for t in tr
            mul!(IBC, x̄ΦI, @view Ω̂tk[:,t])
            @views mul!(Ipk, WpI, ptk[:,t])
            for (r,ir) in enumerate(iI)
                Intxtk[r] = dot(@view(xtk[ir,:,t]), IntΦ)
                kinks && (Intxtk[r] += dot(@view(Ixtkδ[ir,:,t]), IntΦb))
            end
            @views AA[:,t] .= dGx_I*(IBC .+ Intxtk .+ Ipk) .+ GΘΘtk[:,t]
        end
    end

    SO.X̂tk = reshape(-_bb_ldiv(luBB, AA[:])[1:n.X*T2],n.X,T2)
end

"""
    compute_ẑtk_dyn!(ẑtk, FO, QX̂tk)

Backward recursion for the X̂tk-induced part of the second-order individual
response (the z-kernels applied to the path QX̂tk, without storing kernels):
    ẑtk_t = fFze·E_f[Sez ẑtk_{t+1} + SeX QX̂tk_{t+1}] + fF̃zQ·QX̂tk_t.
`QX̂tk` may have fewer than T columns (compute_XZZjk); missing ones are 0.
"""
function compute_ẑtk_dyn!(ẑtk::Array{Tf,3}, FO::FirstOrderApproximation{Tf},
                           QX̂tk::AbstractMatrix) where {Tf}
    ZO = FO.ZO
    n = ZO.n
    T2 = size(ẑtk, 3)
    xe   = zeros(Tf, n.e, n.sp)
    ybuf = zeros(Tf, n.ye, n.sp)
    Eye  = zeros(Tf, n.ye, n.sp)
    zQ   = zeros(Tf, n.Q)
    for t in T2:-1:1
        if t < T2
            zQ .= t+1 <= size(QX̂tk,2) ? @view(QX̂tk[:,t+1]) : zero(Tf)
            reconstruct_rows!(xe, FO.Sez, FO.SeX, view(ẑtk,:,:,t+1), zQ)
            expect_f!(Eye, xe, ybuf, ZO)
        else
            Eye .= zero(Tf)
        end
        zQ .= t <= size(QX̂tk,2) ? @view(QX̂tk[:,t]) : zero(Tf)
        threaded_foreach(n.sp) do _, jr
            for j in jr
                z_j = @view ẑtk[:,j,t]
                @views mul!(z_j, FO.fFze[:,:,j], Eye[:,j])
                @views mul!(z_j, FO.fF̃zQ[:,:,j], zQ, one(Tf), one(Tf))
            end
        end
    end
    return ẑtk
end

"""
    compute_xZZ!(SO)

Full path of the second order derivatives x_ZZ (full_x runs), including the
Ω̂tk correction for the X̂tk-induced savings and probability changes.
"""
function compute_xZZ!(SO::SecondOrderApproximation{Tf}) where {Tf}
    @unpack FO, xtk, X̂tk = SO
    @unpack ZO, T, M, Mp = FO
    @unpack Q, n, Λ = ZO
    na, nc = n.a, n.c
    QX̂tk = Q*X̂tk

    size(SO.ẑtk_ws) == (n.z, n.sp, T) || (SO.ẑtk_ws = Array{Tf,3}(undef, n.z, n.sp, T))
    ẑtk = SO.ẑtk_ws
    compute_ẑtk_dyn!(ẑtk, FO, QX̂tk)

    x̂tk = SO.x̂tk = zeros(Tf,n.x,n.sp,T)
    Ω̂tk_t = zeros(Tf, n.Ω)   #correction for the Ω̂tk term
    Ω̂tk_t_ = zeros(Tf, n.Ω)
    zQ = zeros(Tf, n.Q)
    âbuf = zeros(Tf, n.spr)
    xe = zeros(Tf, n.e, n.sp)
    p̂dyn = zeros(Tf, n.spr)
    for t in 1:T
        zQ .= t <= size(QX̂tk,2) ? @view(QX̂tk[:,t]) : zero(Tf)
        x̂tk_t = @view(x̂tk[:,:,t])
        reconstruct_rows!(x̂tk_t, FO.Wz, FO.WX, view(ẑtk,:,:,t), zQ)
        x̂tk_t .+= @view(xtk[:,:,t])
        if t < T
            @doinplace Ω̂tk_t .= Λ*Ω̂tk_t_
            for c in 1:nc, d in 1:na
                âbuf .= @view ẑtk[d,(c-1)*n.spr+1:c*n.spr,t]   # only the QX̂tk-induced savings
                @doinplace Ω̂tk_t .+= M[d,c]*âbuf
            end
            reconstruct_rows!(xe, FO.Sez, FO.SeX, view(ẑtk,:,:,t), zQ)
            choice_prob!(p̂dyn, xe, ZO)
            @doinplace Ω̂tk_t .+= Mp*p̂dyn
            SO.Ω̂tk[:,t+1] .+= Ω̂tk_t
            Ω̂tk_t, Ω̂tk_t_ = Ω̂tk_t_, Ω̂tk_t
        end
    end
    SO.x̂tk1 = Matrix(x̂tk[:,:,1])
end

"""
    compute_xZZ0!(SO)

Only the initial period of x_ZZ (sufficient for the σσ terms).
"""
function compute_xZZ0!(SO::SecondOrderApproximation{Tf}) where {Tf}
    @unpack FO, xtk, X̂tk = SO
    @unpack ZO, T = FO
    @unpack Q, n = ZO
    QX̂tk = Q*X̂tk
    T2 = size(QX̂tk, 2)

    size(SO.ẑtk_ws) == (n.z, n.sp, T2) || (SO.ẑtk_ws = Array{Tf,3}(undef, n.z, n.sp, T2))
    ẑtk = SO.ẑtk_ws
    compute_ẑtk_dyn!(ẑtk, FO, QX̂tk)

    x̂tk1 = SO.x̂tk1 = zeros(Tf, n.x, n.sp)
    reconstruct_rows!(x̂tk1, FO.Wz, FO.WX, view(ẑtk,:,:,1), view(QX̂tk,:,1))
    x̂tk1 .+= @view xtk[:,:,1]
end


"""
    compute_ZZ_transition!(SO, inputs; full_x=false)

The complete (t,k) interaction chain for ONE pair of directions already set
on SO/FO (FO.ẑt/X̂t/Ω̂t/X_0/Θ_0 and SO.ẑk/X̂k/Ω̂k/X_02/Θ_02): Lemma 2, Lemma 3
(individual block + kinks), Corollary 2 (distribution), Proposition 1 (X̂tk)
and the individual paths.  Used by the deterministic transition experiment
(both directions = the initial-condition direction) and by
computeSecondOrder!.
"""
function compute_ZZ_transition!(SO::SecondOrderApproximation{Tf}, inputs::Inputs; full_x::Bool=false) where {Tf}
    ZO = SO.FO.ZO
    !isempty(ZO.ℵ) && size(SO.Φb.Φs[1], 1) == 1 && ((SO.Φb, SO.Φbr) = construct_Φb(ZO))
    compute_Lemma2_ZZ_AD!(SO, inputs)
    compute_lemma3_ZZ_AD_inplace!(SO, inputs)
    compute_lemma3_ZZ_kink!(SO)
    compute_Corollary2_ZZ!(SO)
    compute_XZZ!(SO)
    full_x ? compute_xZZ!(SO) : compute_xZZ0!(SO)
    return SO.X̂tk
end


"""
    computeSecondOrder!(SO, inputs; full_x=false, diagonal_only=false)

Computes the second-order (t,k) interactions for the aggregate shocks
(no σσ terms).  AD-only: `inputs` is REQUIRED.  Stores the t=1 interaction
slices (x̂_ΘΘ1 / Ixtkδ_ΘΘ1); full x̂/Ω̂ paths only under full_x.
`diagonal_only` skips the off-diagonal (i,j) shock pairs.
Requires compute_Θ_derivatives! and compute_x_Θ_derivatives! on FO.
"""
function computeSecondOrder!(SO::SecondOrderApproximation{Tf}, inputs::Inputs; full_x=false,
                             diagonal_only::Bool=false) where {Tf}
    @unpack FO = SO
    @unpack ZO = FO
    @unpack n = ZO

    diagonal_only && @assert isdiag(ZO.Σ_Θ) "diagonal_only is only exact for a diagonal Σ_Θ"

    !isempty(ZO.ℵ) && ((SO.Φb, SO.Φbr) = construct_Φb(ZO))

    SO.X̂_ΘΘ = Matrix{Matrix{Tf}}(undef,n.Θ,n.Θ)
    SO.x̂_ΘΘ1 = Matrix{Matrix{Tf}}(undef,n.Θ,n.Θ)
    SO.Ixtkδ_ΘΘ1 = Matrix{Matrix{Tf}}(undef,n.Θ,n.Θ)
    full_x && (SO.x̂_ΘΘ = Matrix{Array{Tf,3}}(undef,n.Θ,n.Θ))
    full_x && (SO.Ω̂_ΘΘ = Matrix{Matrix{Tf}}(undef,n.Θ,n.Θ))
    SO.X_02 = zeros(Tf,n.A)
    FO.X_0 = zeros(Tf,n.A)
    FO.Ω_0 = zeros(Tf,n.Ω)

    ZERO_X = diagonal_only ? zeros(Tf, n.X, FO.T) : zeros(Tf, 0, 0)
    ZERO_1 = diagonal_only ? zeros(Tf, n.x, n.sp) : zeros(Tf, 0, 0)
    ZERO_x = diagonal_only ? zeros(Tf, n.x, n.sp, 1) : zeros(Tf, 0, 0, 0)
    ZERO_Ω = diagonal_only ? zeros(Tf, size(FO.Ω̂_Θt[1])...) : zeros(Tf, 0, 0)

    for i in 1:n.Θ
        FO.Θ_0 = I[1:n.Θ,i]
        FO.ẑt = FO.ẑ_Θt[i]
        FO.X̂t = FO.X̂_Θt[i]
        FO.Ω̂t = FO.Ω̂_Θt[i]

        for j in i:n.Θ
            if diagonal_only && ZO.Σ_Θ[i,j] == 0.0
                SO.X̂_ΘΘ[i,j]      = SO.X̂_ΘΘ[j,i]      = ZERO_X
                SO.x̂_ΘΘ1[i,j]     = SO.x̂_ΘΘ1[j,i]     = ZERO_1
                SO.Ixtkδ_ΘΘ1[i,j] = SO.Ixtkδ_ΘΘ1[j,i] = ZERO_1
                if full_x
                    SO.x̂_ΘΘ[i,j] = SO.x̂_ΘΘ[j,i] = ZERO_x
                    SO.Ω̂_ΘΘ[i,j] = SO.Ω̂_ΘΘ[j,i] = ZERO_Ω
                end
                continue
            end
            SO.Θ_02 = I[1:n.Θ,j]
            SO.ẑk = FO.ẑ_Θt[j]
            SO.X̂k = FO.X̂_Θt[j]
            SO.Ω̂k = FO.Ω̂_Θt[j]
            compute_Lemma2_ZZ_AD!(SO, inputs)
            compute_lemma3_ZZ_AD_inplace!(SO, inputs)
            compute_lemma3_ZZ_kink!(SO)
            compute_Corollary2_ZZ!(SO)
            compute_XZZ!(SO)
            if full_x
                compute_xZZ!(SO)
            else
                compute_xZZ0!(SO)
            end
            SO.X̂_ΘΘ[i,j] = SO.X̂tk
            SO.X̂_ΘΘ[j,i] = SO.X̂tk
            SO.x̂_ΘΘ1[i,j] = SO.x̂tk1
            SO.x̂_ΘΘ1[j,i] = SO.x̂tk1
            Iδ1 = Matrix(@view SO.Ixtkδ[:,:,1])
            SO.Ixtkδ_ΘΘ1[i,j] = Iδ1
            SO.Ixtkδ_ΘΘ1[j,i] = Iδ1
            if full_x
                SO.x̂_ΘΘ[i,j] = SO.x̂tk
                SO.x̂_ΘΘ[j,i] = SO.x̂tk
                Ω̂tk_copy = copy(SO.Ω̂tk)
                SO.Ω̂_ΘΘ[i,j] = Ω̂tk_copy
                SO.Ω̂_ΘΘ[j,i] = Ω̂tk_copy
            end
        end
    end
end


function compute_XZZjk(SO::SecondOrderApproximation{Tf},iΘ,jΘ,Tjs,inputs::Inputs) where {Tf}
    @unpack FO = SO
    @unpack ZO, T = FO
    @unpack P, n, ρ_Θ = ZO

    !isempty(ZO.ℵ) && ((SO.Φb, SO.Φbr) = construct_Φb(ZO))

    FO.Θ_0 = I[1:n.Θ,iΘ]
    FO.ẑt  = FO.ẑ_Θt[iΘ]
    FO.X̂t  = FO.X̂_Θt[iΘ]
    FO.Ω̂t  = FO.Ω̂_Θt[iΘ]

    X̂tk_ijtemp = Vector{Matrix{Tf}}(undef,maximum(Tjs))
    for j in Tjs
        SOtemp = copy(SO)
        SOtemp.ẑk = FO.ẑ_Θt[jΘ][:,:,j+1:end]
        SOtemp.X̂k = FO.X̂_Θt[jΘ][:,j+1:end]
        SOtemp.X_02 = P*FO.X̂_Θt[jΘ][:,j]
        SOtemp.Θ_02 = ρ_Θ^j*I[1:n.Θ,jΘ]
        SOtemp.Ω̂k = FO.Ω̂_Θt[jΘ][:,j+1:end]
        compute_Lemma2_ZZ_AD!(SOtemp, inputs)
        compute_lemma3_ZZ_AD_inplace!(SOtemp, inputs)
        compute_lemma3_ZZ_kink!(SOtemp)
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
