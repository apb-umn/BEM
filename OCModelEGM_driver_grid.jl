
include("OCModelEGMInputs.jl")
include("OCModelEGM.jl")

momfilename="base_moments.tex"
momfilename_grid="base_moments_grid.tex"

OCM=OCModel()
OCM.Ia = 2000
setup!(OCM)
OCM.ibise = 0
OCM.rlb=0.038677701422384594*.8
OCM.rub=0.038677701422384594*1.2
OCM.trlb =0.4680612510778516*.8
OCM.trub =0.4680612510778516*1.2
ichk = 0
ss,lev,shr,res = solvess!(OCM)
updatecutoffs!(OCM)
df = compare_moments_grid(OCM)


#Do some Plots
ah = OCM.alθ[1:OCM.Ia,1]

#Compare consumption policy functions

#policy for workers
a_polw,c_polw,λ_polw,v_polw = policyw_gridsearch(OCM)
plot(layout=(2,2),linewidth=2,color=:black)
plot!(subplot=1,ah,OCM.wf.c[1](ah),linestyle=:solid,color=:black,label="EGM")
plot!(subplot=1,ah,c_polw[:,1],linestyle=:dash,color=:black,label="Grid Search")
title!(subplot=1,"Low ϵ Low z")
plot!(subplot=2,ah,OCM.wf.c[5](ah),linestyle=:solid,color=:black,label="EGM")
plot!(subplot=2,ah,c_polw[:,5],linestyle=:dash,color=:black,legend=false)
title!(subplot=2,"Low ϵ High z")
plot!(subplot=3,ah,OCM.wf.c[21](ah),linestyle=:solid,color=:black,label="EGM")
plot!(subplot=3,ah,c_polw[:,21],linestyle=:dash,color=:black,legend=false)
title!(subplot=3,"High ϵ Low z")
plot!(subplot=4,ah,OCM.wf.c[25](ah),linestyle=:solid,color=:black,label="EGM")
plot!(subplot=4,ah,c_polw[:,25],linestyle=:dash,color=:black,legend=false)
title!(subplot=4,"High ϵ High z")

#policy for business owners 

c_polb,a_polb,k_polb,n_polb,y_polb,π_polb,λ_polb,v_polb = policyb_gridsearch(OCM)
range = 1:OCM.Ia-10
ah = ah[range]
plot(layout=(2,2),linewidth=2,color=:black)
plot!(subplot=1,ah,OCM.bf.c[1](ah),linestyle=:solid,color=:black,label="EGM")
plot!(subplot=1,ah,c_polb[range,1],linestyle=:dash,color=:black,label="Grid Search")
title!(subplot=1,"Low ϵ Low z")
plot!(subplot=2,ah,OCM.bf.c[5](ah),linestyle=:solid,color=:black,label="EGM")
plot!(subplot=2,ah,c_polb[range,5],linestyle=:dash,color=:black,legend=false)
title!(subplot=2,"Low ϵ High z")
plot!(subplot=3,ah,OCM.bf.c[21](ah),linestyle=:solid,color=:black,label="EGM")
plot!(subplot=3,ah,c_polb[range,21],linestyle=:dash,color=:black,legend=false)
title!(subplot=3,"High ϵ Low z")
plot!(subplot=4,ah,OCM.bf.c[25](ah),linestyle=:solid,color=:black,label="EGM")
plot!(subplot=4,ah,c_polb[range,25],linestyle=:dash,color=:black,legend=false)
title!(subplot=4,"High ϵ High z")


using CSV, DataFrames, Plots, PGFPlotsX, LaTeXStrings
# Ensure output folders exist

# ---------- Workers ----------
# existing:
 ah = OCM.alθ[1:OCM.Ia,1]
# a_polw,c_polw,λ_polw,v_polw = policyw_gridsearch(OCM)

W = DataFrame(
    a        = ah,
    egm_ll   = OCM.wf.c[1](ah),   grid_ll = c_polw[:, 1],   # Low ε, Low z
    egm_lh   = OCM.wf.c[5](ah),   grid_lh = c_polw[:, 5],   # Low ε, High z
    egm_hl   = OCM.wf.c[21](ah),  grid_hl = c_polw[:, 21],  # High ε, Low z
    egm_hh   = OCM.wf.c[25](ah),  grid_hh = c_polw[:, 25]   # High ε, High z
)
CSV.write("grid_egm_policy_workers.csv", W)

# ---------- Business Owners ----------
# existing:
# c_polb,a_polb,k_polb,n_polb,y_polb,π_polb,λ_polb,v_polb = policyb_gridsearch(OCM)
range  = 1:OCM.Ia-10
ah_b   = ah[range]

B = DataFrame(
    a        = ah_b,
    egm_ll   = OCM.bf.c[1](ah_b),   grid_ll = c_polb[range, 1],
    egm_lh   = OCM.bf.c[5](ah_b),   grid_lh = c_polb[range, 5],
    egm_hl   = OCM.bf.c[21](ah_b),  grid_hl = c_polb[range, 21],
    egm_hh   = OCM.bf.c[25](ah_b),  grid_hh = c_polb[range, 25]
)
CSV.write("grid_egm_policy_business.csv", B)

# (Optional) You also computed df = compare_moments_grid(OCM)
# If you want that too:
# CSV.write("data/compare_moments_grid.csv", df)

