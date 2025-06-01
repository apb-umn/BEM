

include("OCDriverE_transition.jl")   # defines compute_FO_transition_path
# === Simulation Setup ===

# --- Old steady state ---
OCM_ = OCModel()
r_val,tr_val = 0.04162616198911325, 0.6221974322920143
assign!(OCM_, r_val, tr_val)
inputs_ = construct_inputs(OCM_)

# Extract initial asset distribution and capital
X̄_ = getX(OCM_)
A_0 = X̄_[inputs_.Xlab .== :A][1]                       # Initial capital stock
ω̄_0_base = sum(reshape(OCM_.ω, :, 2), dims=2)         # Distribution over (a, θ)

# --- New steady state ---
τb_val,τw_val = 0.25,0.4
r_val, tr_val = 0.04175211358784511, 0.636815508510632
# Compute transition path from old to new policy regime
Xpath, Vinit = compute_FO_transition_path(
    τb_val, τw_val, r_val, tr_val, ω̄_0_base, A_0,X̄_;
    T = 300
)

OCM = OCModel()
OCM.τb = τb_val
OCM.τw = τw_val
assign!(OCM, r_val, tr_val)


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

