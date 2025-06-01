# test_ss
include("OCModelE.jl")               # defines OCModel and solvecase
path = "/Users/apb/Dropbox/optimal_business_taxation/noncompliance/Approximation Code/"

df = CSV.read(path*"grid_results_with_values.csv", DataFrame)
df=df[df.diffv.<1e-2,:]

# specify τb and τw values
τb_val, τw_val = 0.25, 0.4
r_val, tr_val=guess_from_csv(τb_val, τw_val, df)
OCM = OCModel()
OCM.τb = τb_val
OCM.τw = τw_val
assign!(OCM, r_val, tr_val)

