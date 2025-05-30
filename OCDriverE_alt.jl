# OCDriverE.jl
#
#   Three business sectors:
#     -- Nonfinancial corporate sector:
#
#         vc(Kc) =  max {(1-τd)dc +(1+γ)/(1+r) vc(Kc')}
#                  Nc,Xc 
#
#           s.t. (1+γ)Kc'=(1-δ)Kc+Xc
#                     Yc = Θ̄ Kc^α Nc^(1-α)
#                     dc = Yc-w*Nc-Xc-τp(Yc-w*Nc-δ*Kc)
#
#    -- Financial corporate sector
#         vi(x) =  max  {di+ (1+γ)/(1+r) vi(x')} 
#                   x'    
#
#           s.t. di = q*s+b+∫kb-∫a 
#                      +(1-τd)dc*s+r*b-rp∫kb-r∫a (income)
#                      -(1+γ)[q*s'+b'-∫a']-∫[kb'-(1-δ)kb]
#                   = net worth+income-invetment
#
#        ==> r=rp-δ=(1-τd)*[q+(1-τd)dc]/q[-1]-1
#                =(1-τp)(α*Yc/Kc-δ)
#
#    -- Private sector operating DRS technology 
#
#         vb(a,θw,θb) =  max U(c) + β E v(a',θw',θb',η')
#                        c,a'
#
#           s.t. c+a' = (1+r)a + π - τb*π-τc*c
#                π = max θb kb^α nb^ν-(r+δ)*kb-w*nb
#
#   Workers:
#
#         vw(a,θw,θb) =  max U(c) + β E v(a',θw',θb',η')
#                       c,a'
#
#           s.t. c+a' = (1+r)*a + w*θw -τw*w*θw -τc*c
#
#   Occupation choice:
#
#         η = taste shock = ηw-ηb ~ logistic
#         p = probability of being a worker
#         v = max d*(vw+η) + (1-d)*vb
#
#        Ev = 0.57722*σ_η + σ_η *log(exp(vw/σ_η)+exp(vb/σ_η))
#           = 0.57722*σ_η + vw+ σ_η*(1+exp((vb-vw)/σ_η))
#
#   Market clearing:
#
#        Kc + ∫ (1-d[i])k[i] di = ∫ a[i] di
#        Nc + ∫ (1-d[i])n[i] di = ∫ d[i] θw[i] di
#
#   Government budget:
#
#        G+Tr+r*B =B'-B+τc*C+τw*w*(Nc+Nb)
#                  +τp*(Yc-w*Nc-δ*Kc)
#                  +τd*(Yc-w*Nc-Xc-τp*(Yc-w*Nc-δ*Kc))
#                  +τb*(Yp-(r+δ)*Kp-w*Nb)
#

#   Ellen McGrattan, 5/9/2025
#   Revised, ERM, 5/25/2025


include("OCModelE_alt.jl")
results_df = CSV.read("grid_results.csv", DataFrame)
τb_val=0.488
τw_val=0.266
OCM=OCModel()
OCM.Na=30
OCM.so=2
OCM.τb = τb_val
OCM.τw = τw_val

function guess_from_csv(τb, τw, df)
    row = df[(df.τb .≈ τb) .& (df.τw .≈ τw), :]
    if nrow(row) == 1 && !isnan(row.r[1]) && !isnan(row.tr[1])
        return row.r[1], row.tr[1]
    else
        # Prepare data: each column is a point in 2D
        pts = permutedims(hcat(df.τb, df.τw))  # 2×N matrix
        query = reshape([τb, τw], 2, 1)        # 2×1 matrix
        dists = pairwise(Euclidean(), pts, query)  # (N×1 matrix)
        idx = argmin(dists)
        return df.r[idx], df.tr[idx]
    end
end
r_guess, tr_guess = guess_from_csv(τb_val, τw_val, results_df)
assign!(OCM, r_guess, tr_guess)

# check the marginal value of wealth

function get_marginal_value_wealth(OCM::OCModel)
        @unpack Vcoefs,σ,β,γ,Nθ,lθ,a̲,EΦ_aeg,Na,agrid,α_b,ν,δ,χ,r,w,tr,τc,τb = OCM
    
        #Compute value function derivative
        EVₐ′ = reshape(EΦ_aeg*Vcoefs,:,Nθ) 
        return EVₐ′
end

EVₐ′= get_marginal_value_wealth(OCM) 


function get_marginal_value_wealth_analytical(OCM::OCModel)


    @unpack bf,wf,Nθ,agrid,Na,σ_ε,σ,r,EΦeg,Φ =OCM

    c_w,c_b = zeros(Na,Nθ),zeros(Na,Nθ) 
    Vw,Vb   = zeros(Na,Nθ),zeros(Na,Nθ)    
    Vw_a,Vb_a   = zeros(Na,Nθ),zeros(Na,Nθ)



    cf_w=wf.c 
    vf_w=wf.v
    cf_b=bf.c
    vf_b=bf.v


    for s in 1:Nθ
        c_w[:,s] = cf_w[s](agrid) 
        Vw[:,s]  = vf_w[s](agrid)
        c_b[:,s] = cf_b[s](agrid)
        Vb[:,s]  = vf_b[s](agrid)
        Vw_a[:,s] =  c_w[:,s].^(-σ)*(1+r)
        Vb_a[:,s] =  c_b[:,s].^(-σ)*(1+r)
    end

      p       = probw.(Vb.-Vw,σ_ε)
      V       = p.*Vw .+ (1 .- p).*Vb
      ptol    = 1e-8
      ip      = ptol.< p .< 1-ptol
      V[ip]  .= Vw[ip] .+ σ_ε.*log.(1 .+ exp.((Vb[ip].-Vw[ip])./σ_ε))
    Vₐ′ = p.*Vw_a .+ (1 .- p).*Vb_a
    luΦ   = lu(Φ)
    Vacoefs   = luΦ\Vₐ′[:]
    EVₐ′ = reshape(EΦeg*Vacoefs,:,Nθ) 

    return EVₐ′
end

EVₐ′_analytical = get_marginal_value_wealth_analytical(OCM)

# plot the two sets of marginal values
using Plots
s=18

agrid= OCM.agrid
plot(agrid, EVₐ′[:,s], label="Numerical", xlabel="a", ylabel="Marginal Value of Wealth", title="Marginal Value of Wealth for s=$s")
plot!(agrid, EVₐ′_analytical[:,s], label="Analytical", linestyle=:dash)

# zoom for agrid < 2
sel= agrid .< 4. # select a range for plotting
plot(agrid[sel], EVₐ′[sel,s], label="Numerical", xlabel="a", ylabel="Marginal Value of Wealth", title="Marginal Value of Wealth for s=$s (zoomed in)")
plot!(agrid[sel], EVₐ′_analytical[sel,s], label="Analytical", linestyle=:dash)
# save the plot


s_vals = [1, 4, 13, 18]
titles = ["s = $s" for s in s_vals]

# Optional: zoom range
sel = agrid .< 5.0

# 2×2 layout
p = plot(layout = (2, 2), size = (1000, 800))

for (i, s) in enumerate(s_vals)
    plot!(
        agrid[sel], EVₐ′[sel, s],
        label = "Numerical",
        xlabel = "a", ylabel = "Marginal Value",
        title = titles[i],
        legend = :topright,
        subplot = i
    )
    plot!(
        agrid[sel], EVₐ′_analytical[sel, s],
        label = "Analytical",
        linestyle = :dash,
        subplot = i
    )
end

display(p)
# savefig(p, "marginal_value_subplots.pdf")
