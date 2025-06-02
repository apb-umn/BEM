

include("OCDriverE_transition.jl")   # defines compute_FO_transition_path
path= "/Users/bhandari/Dropbox/optimal_business_taxation/noncompliance/Approximation Code/"
results_df = CSV.read(path*"grid_results_with_values.csv", DataFrame)
results_df = filter(row -> all(!isnan, [row.r, row.tr, row.value]), results_df)

# === Simulation Setup ===

# --- Old steady state ---
OCM_ = OCModel()
OCM_.Na =50
r_val,tr_val = 0.041661030621081736, 0.6274225146075384
assign!(OCM_, r_val, tr_val)
OCM_.ibise=0
solvess!(OCM_)
inputs_ = construct_inputs(OCM_)

# Extract initial asset distribution and capital
X̄_ = getX(OCM_)
A_0 = X̄_[inputs_.Xlab .== :A][1]                       # Initial capital stock
ω̄_0_base = sum(reshape(OCM_.ω, :, 2), dims=2)         # Distribution over (a, θ)

# --- New steady state ---
#τb_val,τw_val = 0.25,0.4
#r_val, tr_val = 0.041813949055748666, 0.6536396110456242
τb_val,τw_val = 0.622222,0.622222
r_val, tr_val = guess_from_csv(τb_val, τw_val, results_df)
OCM = deepcopy(OCM_)
OCM.τb = τb_val
OCM.τw = τw_val
assign!(OCM, r_val, tr_val)
OCM.ibise=0
_, _, _, res = solvess!(OCM)
diff_v = diffegm(OCM)
Xss = getX(OCM)  # [R, W, Tr, Frac_b, V, A, C]
R, W, Tr, Frac_b, V, A, C = Xss
# Construct approximation objects around new steady state
inputs = construct_inputs(OCM)
ZO = ZerothOrderApproximation(inputs)
computeDerivativesF!(ZO, inputs)
computeDerivativesG!(ZO, inputs)
FO = FirstOrderApproximation(ZO, OCM.T)

# Compute linear transition system (some functions redundant by design)
compute_f_matrices!(FO)
compute_Lemma3!(FO)
compute_Lemma4!(FO)
compute_Corollary2!(FO)
compute_Proposition1!(FO)
compute_BB!(FO)

# Adjust initial distribution using occupational choices implied by new τb
ω̄ = reshape(OCM.ω, :, 2)
p̄ = ω̄ ./ sum(ω̄, dims=2)
p̄[isnan.(p̄[:, 1]), 1] .= 1.0
p̄[isnan.(p̄[:, 2]), 2] .= 0.0
ω̄_0 = (p̄ .* ω̄_0_base)[:]  # full initial distribution over (a, θ, occupation)

# Set initial deviations from new steady state
FO.X_0 = [A_0] - ZO.P * ZO.X̄
FO.Θ_0 = [0.0]
FO.Δ_0 = ω̄_0 - ZO.ω̄

# Solve forward transition path
solve_Xt!(FO)

# Construct final time path and value function at t=0
Xpath = [X̄_ ZO.X̄ .+ FO.X̂t]
Vinit = ZO.X̄[inputs.Xlab .== :V] + FO.X̂t[inputs.Xlab .== :V, 1]

data=NamedTuple{
            (:τb, :τw, :r, :tr, :diffv, :diffasset, :diffgbc, :Rss, :Wss, :Trss, :Frac_bss, :Vss, :Ass, :Css,:value)
        }((τb_val, τw_val, OCM.r, OCM.tr, diff_v, res[1], res[2], R, W, Tr, Frac_b, V, A, C,Vinit[1]))

function push_if_unique_and_valid!(df::DataFrame, data::NamedTuple, X̂t::AbstractMatrix)
    # Condition 1: No duplicate row in df
    is_duplicate = any(row -> row.τb == data.τb && row.τw == data.τw, eachrow(df))

    # Condition 2: Matrix condition
    is_matrix_valid = all(abs.(X̂t) .< 1e3)

    # Push only if both hold
    if !is_duplicate && is_matrix_valid
        push!(df, data)
    end
end
push_if_unique_and_valid!(results_df, data, FO.X̂t)

CSV.write(path*"grid_results_with_values.csv", df)



# --- Plots ---


df = DataFrame(Xpath',inputs_.Xlab)
df.t = 0:(size(Xpath,2)-1)

default(linewidth=2)
p1 = plot(df.t, df.A, ylabel="Capital", label="")
p2 = plot(df.t, df.Frac_b, ylabel="Fraction Self Employed", label="")
plot(p1, p2, layout=(2, 1), size=(800, 600), legend=:topright)



default(linewidth=2)
p1 = plot(df.t, df.R, ylabel="R", label="")
p2 = plot(df.t, df.Tr, ylabel="Tr", label="")
plot(p1, p2, layout=(2, 1), size=(800, 600), legend=:topright)
# --- Save results ---
CSV.write("df_transition.csv", df)
T=length(df.t)-1
rT        = df.R[2:T+1].-1
trT       = df.Tr[2:T+1]
x0        = vcat(rT,trT)
OCM_.T     = T
OCM.T  = T

res       = residual_tr!(x0,OCM_,OCM)

plot(reshape(res,(T,2)), label="Residuals", xlabel="Time", ylabel="Residual Value")



    # new ss

# results_df = CSV.read("grid_results.csv", DataFrame)
# results_df[!,:value]=ones(length(results_df.r))*NaN
# using Base.Threads

# n = nrow(results_df)
# values = Vector{Union{Float64, Missing}}(undef, n)

# Threads.@threads for i in 1:n
#     row = results_df[i, :]
#     if !ismissing(row.r) && !ismissing(row.tr)
#         try
#             println("Thread $(threadid()) processing row $i")
#             Xpath, Vinit = compute_FO_transition_path(
#                 row.τb, row.τw, row.r, row.tr, ω̄_0_base, A_0,X̄_; T=300)
#             values[i] = Vinit[1]
#         catch e
#             @warn "Error in row $i on thread $(threadid()): $e"
#             values[i] = missing
#         end
#     else
#         values[i] = missing
#     end
# end

# # Assign the computed values to a new column
# results_df.value = values
# CSV.write("grid_results_with_values.csv", results_df)


# save CSV
#CSV.write("df_transition.csv", df)
# 

