

include("OCModelE_jacobian.jl")   # defines compute_FO_transition_path


τb_val=0.25
τw_val=0.4
jac,OCM_,OCM = compute_FO_transition_path(τb_val, τw_val)

default(linewidth=2)
p1 = plot(df.t, df.A, ylabel="Capital", label="")
p2 = plot(df.t, df.Tr, ylabel="Transfers", label="")
plot(p1, p2, layout=(2, 1), size=(800, 600), legend=:topright)

