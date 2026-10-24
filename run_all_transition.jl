include("OCModelEGMInputs.jl")
include("OCModelEGM.jl")
include("OCModelEGM_transition.jl") # has the F,G,ff and several helper functions


# Define parameters
τb_val = 0.4
ρ_τ_val_fast = 0.0
ρ_τ_val_slow = 0.9

# Create output filenames using interpolation
filenamefast = "df_transition_fast_$(round(τb_val, digits=2)).csv"
filenameslow = "df_transition_slow_$(round(τb_val, digits=2)).csv"
saveplotfilename = "transition_comparison_$(round(τb_val, digits=2)).pdf"
# Run the transition analysis function
df_fast, df_slow = run_transition_analysis(
    τb_val,
    ρ_τ_val_fast,
    ρ_τ_val_slow,
    filenamefast,
    filenameslow,
    saveplotfilename;
    r = rguess,
    tr = trguess
)


# Define parameters (optimal τb of the baseline grid in run_all_opt_chi.jl, Figure3.csv)
τb_val = 0.6111428571428571
ρ_τ_val_fast = 0.0
ρ_τ_val_slow = 0.9

# Create output filenames using interpolation
filenamefast = "df_transition_fast_$(round(τb_val, digits=2)).csv"
filenameslow = "df_transition_slow_$(round(τb_val, digits=2)).csv"
saveplotfilename = "transition_comparison_$(round(τb_val, digits=2)).pdf"
# Run the transition analysis function
df_fast, df_slow = run_transition_analysis(
    τb_val,
    ρ_τ_val_fast,
    ρ_τ_val_slow,
    filenamefast,
    filenameslow,
    saveplotfilename;
    r = rguess,
    tr = trguess
)
