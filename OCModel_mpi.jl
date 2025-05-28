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



τb_vals = range(0.1, stop=0.6, length=20)
τw_vals = range(0.1, stop=0.6, length=20)
grid = collect(Iterators.product(τb_vals, τw_vals))

results_raw = pmap(x -> solvecase(x[1], x[2], r_guess, tr_guess), grid)
results_df = DataFrame(vec(results_raw))  
CSV.write(path*"grid_results.csv", results_df)


# #debug space
# results_df = CSV.read("grid_results.csv", DataFrame)
# τb_val=0.3
# τw_val=0.3
# OCM=OCModel()
# OCM.τb = τb_val
# OCM.τw = τw_val
# sel= results_df.τb .== τb_val .&& results_df.τw .== τw_val
# r_guess = results_df[sel, :r][1]
# tr_guess = results_df[sel, :tr][1]
# assign!(OCM, r_guess, tr_guess)
# OCM.Nit=500
# solvess!(OCM)