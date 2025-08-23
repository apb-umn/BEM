
include("OCModelEGMInputs.jl")
include("OCModelEGM.jl")

momfilename="base_moments.tex"
momfilename_grid="base_moments_grid.tex"

OCM=OCModel()
setup!(OCM)
OCM.ibise = 0
OCM.rlb=0.03866410102334246*.8
OCM.rub=0.03866410102334246*1.2
OCM.trlb =0.5204623083622442*.8
OCM.trub =0.5204623083622442*1.2
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

