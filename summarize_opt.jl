
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
OCM_old.ibise = 0
OCM_old.iprint = 0

# Run optimization and comparison
OCM_opt, comparison_df = setup_and_compare_steady_state!(OCM_old, filenamesuffix)
comparison_df

filenamesuffix = "lowchi"
χval=1.05
# Initialize and set up OCM_old
println("Setting up old steady state (takes a few minutes) ...")
OCM_old = OCModel()
setup!(OCM_old)
# Manual calibration
OCM_old.χ = χval
OCM_old.rlb, OCM_old.rub = OCM_old.r * 0.8, OCM_old.r * 1.2
OCM_old.trlb, OCM_old.trub = OCM_old.tr * 0.8, OCM_old.tr * 1.2
OCM_old.ibise = 0
OCM_old.Θ̄ = OCM_old.Θ̄*0.99
OCM_old.iprint = 0

# Run optimization and comparison
OCM_opt, comparison_df = setup_and_compare_steady_state!(OCM_old, filenamesuffix)
comparison_df



#
filenamesuffix="highchi"
χval=2.0

println("Setting up old steady state (takes a few minutes) ...")
OCM_old = OCModel()
setup!(OCM_old)
OCM_old.χ = χval
OCM_old.rlb=OCM_old.r*0.8
OCM_old.rub=OCM_old.r*1.2
OCM_old.trlb=OCM_old.tr*.8
OCM_old.trub=OCM_old.tr*1.2
OCM_old.ibise=0
OCM.Θ̄=OCM.Θ̄*1.03

# Run optimization and comparison
OCM_opt, comparison_df = setup_and_compare_steady_state!(OCM_old, filenamesuffix)
comparison_df



#
filenamesuffix="elasticity"
include("OCModelEGMHighElasticityInputs.jl")
println("Setting up old steady state (takes a few minutes) ...")
OCM_old = OCModel()
setup!(OCM_old)
OCM_old.ibise=0
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


function build_results_csv(; dir=".", outfile="Results.csv")
    # Helper to extract numeric values by row label and column index
    function pick(df, rowlabel, col)
        idx = findfirst(==(rowlabel), String.(df[:, 1]))
        idx === nothing && error("Row '$rowlabel' not found in file")
        x = df[idx, col]
        return x isa AbstractString ? parse(Float64, x) : Float64(x)
    end

    # Input files and order (corresponding to Rows 1–5)
    rows = [
        ("Baseline",             "moments_comparison_base.csv"),
        ("Tighter collateral",   "moments_comparison_highchi.csv"),
        ("Looser collateral",    "moments_comparison_lowchi.csv"),
        ("Higher tax elasticity","moments_comparison_elasticity.csv"),
        ("Higher income risk",   "moments_comparison_highrisk.csv"),
    ]

    # Preallocate storage
    Results = Matrix{Float64}(undef, length(rows), 8)

    # Read files and fill rows
    for (i, (econ, file)) in enumerate(rows)
        df = CSV.read(joinpath(dir, file), DataFrame; header=false)
        rename!(df, [:label, :col2, :col3, :col4])

        # Fill numeric columns
        Results[i, 1] = pick(df, "C (consumption)", 4)
        Results[i, 2] = pick(df, "total. assets (A)", 4)
        Results[i, 3] = pick(df, "wage (w)", 4)
        Results[i, 4] = pick(df, "govt transfer (tr)", 4)
        Results[i, 5] = pick(df, "Fraction of biz owners", 4)
        Results[i, 6] = pick(df, "profits (Pib)", 4)
        Results[i, 7] = pick(df, "tax on biz profits (Tb)", 3)   # add third column if you need both
        Results[i, 8] = pick(df, "CE welfare gains", 4)
    end

    # --- Write manually in your requested numeric layout ---
    open(joinpath(dir, outfile), "w") do io
        println(io, "Row,Column1,Column2,Column3,Column4,Column5,Column6,Column7,Column8")
        for i in 1:size(Results, 1)
            vals = [@sprintf("%.4f", Results[i, j]) for j in 1:8]
            println(io, string(i, ", ", join(vals, ", ")))
        end
    end

    println("✅ Results.CSV written in numeric matrix format to $(joinpath(dir, outfile))")
    return Results
end


# Example usage:
 build_results_csv(dir=".", outfile="Results.csv")
