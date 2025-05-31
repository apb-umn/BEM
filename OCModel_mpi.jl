using Distributed
addprocs(20)  

path="/Users/bhandari/Dropbox/optimal_business_taxation/noncompliance/Approximation Code/"
@everywhere begin
    include("OCModelE.jl")
    OCM=OCModel()
    r_guess=0.0416708746736356
    tr_guess=0.6270852509001674
    assign!(OCM, r_guess, tr_guess)
end



τb_vals = range(0.1, stop=0.6, length=10)
τw_vals = range(0.1, stop=0.6, length=10)
grid = collect(Iterators.product(τb_vals, τw_vals))

results_raw = pmap(x -> solvecase(x[1], x[2], r_guess, tr_guess), grid)
results_df = DataFrame(vec(results_raw))  
CSV.write(path*"grid_results.csv", results_df)


# # #debug space
# results_df = CSV.read("grid_results.csv", DataFrame)
# τb_val=0.4947
# τw_val=0.3105
# OCM=OCModel()
# OCM.τb = τb_val
# OCM.τw = τw_val
# sel= results_df.τb .== τb_val .&& results_df.τw .== τw_val
# # r_guess = results_df[sel, :r][1]
# # tr_guess = results_df[sel, :tr][1]
# OCM.Nit=500
# OCM.Na=500
# assign!(OCM, 0.0420, 0.6611)
# solvess!(OCM)




# Plots
using Plots

@unpack bf, Ia,alθ,Nθ = OCM

ah  = alθ[1:Ia,1] #grids are all the same for all shocks
kb     = hcat([bf.k[s](ah) for s in 1:Nθ]...)
aprime = hcat([bf.a[s](ah) for s in 1:Nθ]...)
cb = hcat([bf.c[s](ah) for s in 1:Nθ]...)
vb = hcat([bf.v[s](ah) for s in 1:Nθ]...)

s=10
sel= ah .< 1. # select a range for plotting
plot(ah[sel], kb[sel,s], label="kb", xlabel="a", ylabel="kb", title="Business Capital")
plot(ah[sel], aprime[sel,s], label="aprime", title="Business Capital")
plot(ah[sel], cb[sel,s], label="cb", title="consumption")
plot(ah[sel], vb[sel,s], label="vb", title="value function")



@unpack Vcoefs,σ,β,γ,Nθ,lθ,a̲,EΦ_aeg,Na,agrid,α_b,ν,δ,χ,r,w,tr,τc,τb = OCM

lθb = lθ[:,1]
θb  = exp.(lθb)

#Initialize policy rules for each productivity
cf  = Vector{Spline1D}(undef,Nθ)
af  = Vector{Spline1D}(undef,Nθ)
kf  = Vector{Spline1D}(undef,Nθ)
nf  = Vector{Spline1D}(undef,Nθ)
yf  = Vector{Spline1D}(undef,Nθ)
πf  = Vector{Spline1D}(undef,Nθ)

#Compute firms profit (ignoring constraints)
nbyk = ν*(r+δ)/(α_b*w) 
kvec = @. (w/(ν*θb*nbyk^(ν-1)))^(1/(α_b+ν-1))
πu   = @. θb*kvec^α_b*(nbyk*kvec)^ν-(r+δ)*kvec-w*(nbyk*kvec)

#Compute value function derivative
EVₐ′ = reshape(EΦ_aeg*Vcoefs,:,Nθ) 

asp=OCM.agrid
sel= asp .< 2. # select a range for plotting
plot(asp[sel], EVₐ′[sel,s], label="EVprime", xlabel="a", ylabel="EV", title="Expected Value Function")
