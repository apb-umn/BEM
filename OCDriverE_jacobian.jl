include("OCModelE_jacobian.jl")   # defines compute_FO_transition_path


τb_val=0.30
τw_val=0.4
df,jac,OCM_,OCM = compute_FO_transition_path(τb_val, τw_val)



default(linewidth=2)
p1 = plot(df.t, df.R, ylabel="R", label="")
p2 = plot(df.t, df.Tr, ylabel="Tr", label="")
plot(p1, p2, layout=(2, 1), size=(800, 600), legend=:topright)


T=length(df.t)-1
rT        = df.R[2:T+1].-1
trT       = df.Tr[2:T+1]
x0        = vcat(rT,trT)
solve_tr!(x0,OCM_, OCM)

# dx  = similar(x)                   # Newton step

# ξ=0.1

# tol=1e-5

# x   = copy(x0)                     # preserve original input
# fx  = similar(x)                   # f(x)


# for it in 1:100
#     fx .= residual_tr!(x,OCM_,OCM)                 # res = f(x)
#     if norm(fx) < tol
#         return x
#     end

#     @views begin
#         dx .= jac \ fx              # solve J dx = fx
#         x  .-= ξ .* dx            # update step
#     end
#     println("Iteration $it: norm(fx) = $(norm(fx))")
# end

