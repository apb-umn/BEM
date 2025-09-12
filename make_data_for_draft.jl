# This files constructs the csv files used in the draft.
# 1. Parameters
# 2. productivities
# 3. SS moments
# 4. Figure1: Data for fast (rho_tau=0) transition
# 5. Figure2: Data for slow (rho_tau=0.9) transition
# 6. Figure3: Data for optimal taub
# 7. Figure4: Data Data for fast (rho_tau=0) transition at opt
# 8. Figure5: Data for slow (rho_tau=0.9) transition at opt

include("OCModelEGM_driver.jl")

# 1. Define the parameters for the paper
 export_parameters_for_paper(OCM, "Parameters.csv")
# 2. Define the productivities for the paper
export_BM_tables(OCM, "Productivities.csv")
# 3. Define the steady state moments for the paper
getMoments(OCM, savepath="SSmoments.csv")

# 4. Figure 1: Data for fast transition (ρ_τ = 0)
#include("run_all_transition.jl")
df_fast = CSV.read("df_transition_fast_0.4.csv", DataFrame)
# drop column :Iλ
select!(df_fast, Not(:Iλ))
#save the DataFrame to a CSV file
CSV.write("Figure1.csv", df_fast)

# 5. Figure 2: Data for slow transition (ρ_τ = 0.9)
df_slow = CSV.read("df_transition_slow_0.4.csv", DataFrame)
# drop column :Iλ
select!(df_slow, Not(:Iλ))
#save the DataFrame to a CSV file
CSV.write("Figure2.csv", df_slow)

# 6. Figure 3: Data for optimal taub
# include("OCModel_opttaxmpi_driver.jl")
df_filtered_smooth = CSV.read("data_opt_base_smooth_filtered.csv", DataFrame)
# rename to taub,rhotau,Vss,VinitFO,VinitSO,CESS,CEFO,CESO,VinitSO_smooth
rename!(df_filtered_smooth, :τb => :taub, :ρ_τ => :rhotau, :Vss => :Vss, 
    :VinitFO => :VinitFO, :VinitSO => :VinitSO, :CESS => :CESS, 
    :CEFO => :CEFO, :CESO => :CESO, :VinitSO_smooth => :VinitSO_smooth)
#save the DataFrame to a CSV file
CSV.write("Figure3.csv", df_filtered_smooth)


# 4. Figure 4: Data for fast transition (ρ_τ = 0) for opt
#   include("run_all_transition.jl")
df_fast = CSV.read("df_transition_fast_0.59.csv", DataFrame)
# drop column :Iλ
select!(df_fast, Not(:Iλ))
#save the DataFrame to a CSV file
CSV.write("Figure4.csv", df_fast)

# 5. Figure 5: Data for slow transition (ρ_τ = 0.9) for opt
df_slow = CSV.read("df_transition_slow_0.59.csv", DataFrame)
# drop column :Iλ
select!(df_slow, Not(:Iλ))
#save the DataFrame to a CSV file
CSV.write("Figure5.csv", df_slow)

# create data for robustness exercises
include("summarize_opt.jl")
