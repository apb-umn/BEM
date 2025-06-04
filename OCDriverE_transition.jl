

include("OCModelE_transition.jl")   # defines compute_FO_transition_path


τb_val=0.57
τw_val=0.4
df,sol,OCM_,OCM = compute_FO_transition_path(τb_val, τw_val)



# --- results/debug space ---


default(linewidth=2)
p1 = plot(df.t, df.A, ylabel="Capital", label="")
p2 = plot(df.t, df.Frac_b, ylabel="Fraction Self Employed", label="")
plot(p1, p2, layout=(2, 1), size=(800, 600), legend=:topright)

p3 = plot(df.t, df.W, ylabel="Wage", label="")


p3 = plot(df.t, df.C, ylabel="Consumption", label="")


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

