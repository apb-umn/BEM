using DataFrames

include("OCModelEGMInputs.jl")
include("OCModelEGM.jl")


# --- Turn your 2-column matrix into ordered (name => value) pairs
_moments_as_pairs(moments_mat) = begin
    @assert size(moments_mat, 2) == 2 "Expected a 2-column moments matrix."
    # preserve row order and coerce to (String => Float64)
    [String(moments_mat[i,1]) => float(moments_mat[i,2]) for i in 1:size(moments_mat,1)]
end

function solve_for_sigma!(OCM, σ; rguess, trguess, bounds_scale=0.2, momfile_prefix="temp_moments")
    OCM.σ_ε = σ
    OCM.r   = rguess
    OCM.tr  = trguess
    OCM.rlb = rguess * (1 - bounds_scale)
    OCM.rub = rguess * (1 + bounds_scale)
    OCM.trlb = trguess * (1 - bounds_scale)
    OCM.trub = trguess * (1 + bounds_scale)

    setup!(OCM)
    OCM.ibise = 0
    ss, lev, shr, res = solvess!(OCM)
    updatecutoffs!(OCM)

    momfile = "$(momfile_prefix)_sigma=$(σ).tex"
    moments_mat = getMoments(OCM, savepath=momfile)   # <-- returns your 2-col matrix
    pairs = _moments_as_pairs(moments_mat)

    return pairs, OCM.r, OCM.tr
end

function run_sigma_grid(sigmas::AbstractVector{<:Real};
                        r0=0.03886289537976518,
                        tr0=0.5275187451776279,
                        bounds_scale=0.2,
                        momfile_prefix="temp_moments")

    OCM = OCModel()
    #OCM.Na =500
    OCM.Θ̄ =0.645

    rguess, trguess = r0, tr0

    moment_names = String[]
    cols = Dict{Float64, Vector{Float64}}()
    meta = Dict{Float64, Dict{String, Any}}()

    for (i, σ) in enumerate(sigmas)
        pairs, r_sol, tr_sol = solve_for_sigma!(OCM, σ;
            rguess=rguess, trguess=trguess, bounds_scale=bounds_scale, momfile_prefix=momfile_prefix)

        if i == 1
            moment_names = [p.first for p in pairs]  # lock row order on first run
        end
        col = [Dict(pairs)[name] for name in moment_names]

        cols[float(σ)] = col
        meta[float(σ)] = Dict("r" => r_sol, "tr" => tr_sol)

        rguess, trguess = r_sol, tr_sol   # warm-start next σ
    end

    df = DataFrame(Moment = moment_names)
    for σ in sort!(collect(keys(cols)))
        df[!, string(σ)] = cols[σ]
    end
    return df, meta
end

# --- Example usage
sigmas = [0.0001/2, 0.0001/3, 0.0001/4, 0.0001/5]
df_moments, results = run_sigma_grid(sigmas; momfile_prefix="temp_moments")
