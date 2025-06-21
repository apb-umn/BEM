
include("OCModelEGMInputs.jl")
include("OCModelEGM.jl")

function setup_and_compare_steady_state!(OCM_old::OCModel, filenamesuffix::String)

    # Solve and save baseline steady state
    solvess!(OCM_old)
    updatecutoffs!(OCM_old)
    println("Old steady state setup complete.")

    export_parameters_for_paper(OCM_old, "Parameters"*filenamesuffix*".csv")
    getMoments(OCM_old, savepath="SSmoments"*filenamesuffix*".csv")

    # Read optimal CESO results
    file1 = joinpath(".", "data_opt_"*filenamesuffix*"_smooth_filtered.csv")
    df1 = CSV.read(file1, DataFrame)
    df1 = select(df1, [:τb, :CESO])

    best_idx = argmax(df1.CESO)
    best_taub = df1.τb[best_idx]
    best_ceso = df1.CESO[best_idx]
    println("Best τb for $filenamesuffix: $best_taub")
    println("Best CESO for $filenamesuffix: $best_ceso")

    # Setup optimal model
    OCM_opt = deepcopy(OCM_old)
    OCM_opt.τb = best_taub
    solvess!(OCM_opt)
    updatecutoffs!(OCM_opt)

    # Save moments and parameters for optimal steady state
    getMoments(OCM_opt, savepath="SSmoments_opt"*filenamesuffix*".csv")
    export_parameters_for_paper(OCM_opt, "Parameters_opt"*filenamesuffix*".csv")

        # Compare and save
        comparison_df = compare_moments(OCM_old, OCM_opt)
        # Construct new row
    cerow = DataFrame(
        :Moment => ["CE welfare gains"],
        :Baseline => [0.0],
        :Alternative => [best_ceso],
        Symbol("% Diff") => [best_ceso*100]
    )

    # Append it
    append!(comparison_df, cerow)

    CSV.write("moments_comparison_"*filenamesuffix*".csv", comparison_df)

    return OCM_opt, comparison_df
end



#
filenamesuffix="base"
println("Setting up old steady state (takes a few minutes) ...")
OCM_old = OCModel()
setup!(OCM_old)
OCM_old.r = 0.038374261709122705
OCM_old.tr = 0.6713394785619925
OCM_old.rlb, OCM_old.rub = OCM_old.r * 0.8, OCM_old.r * 1.2
OCM_old.trlb, OCM_old.trub = OCM_old.tr * 0.8, OCM_old.tr * 1.2
OCM_old.ibise = 0
OCM_old.iprint = 0

# Run optimization and comparison
OCM_opt, comparison_df = setup_and_compare_steady_state!(OCM_old, filenamesuffix)
comparison_df

filenamesuffix = "lowchi"
χval = 1.25
# Initialize and set up OCM_old
println("Setting up old steady state (takes a few minutes) ...")
OCM_old = OCModel()
setup!(OCM_old)
# Manual calibration
OCM_old.χ = χval
OCM_old.r, OCM_old.tr = 0.03793062017083158, 0.5050856379159755
OCM_old.rlb, OCM_old.rub = OCM_old.r * 0.8, OCM_old.r * 1.2
OCM_old.trlb, OCM_old.trub = OCM_old.tr * 0.8, OCM_old.tr * 1.2
OCM_old.ibise = 0
OCM_old.Θ̄ = 0.645
OCM_old.iprint = 0

# Run optimization and comparison
OCM_opt, comparison_df = setup_and_compare_steady_state!(OCM_old, filenamesuffix)
comparison_df



#
filenamesuffix="highchi"
χval=3.0

println("Setting up old steady state (takes a few minutes) ...")
OCM_old = OCModel()
setup!(OCM_old)
OCM_old.χ = χval
OCM_old.χ = χval
OCM_old.r,OCM_old.tr=0.03899143754541571, 0.5159833125479533
OCM_old.rlb=OCM_old.r*0.8
OCM_old.rub=OCM_old.r*1.2
OCM_old.trlb=OCM_old.tr*.8
OCM_old.trub=OCM_old.tr*1.2
OCM_old.ibise=0
OCM_old.Θ̄ = 0.65*1.01
OCM_old.iprint = 0

# Run optimization and comparison
OCM_opt, comparison_df = setup_and_compare_steady_state!(OCM_old, filenamesuffix)
comparison_df


include("OCModelEGMHighRiskInputs.jl")
filenamesuffix="highrisk"
println("Setting up old steady state (takes a few minutes) ...")
OCM_old = OCModel()
setup!(OCM_old)
OCM_old.ibise = 0
OCM_old.iprint = 0

# Run optimization and comparison
OCM_opt, comparison_df = setup_and_compare_steady_state!(OCM_old, filenamesuffix)
comparison_df