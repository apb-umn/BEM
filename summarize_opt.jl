
using CSV, DataFrames


function read_and_merge_opt_data(casename::String, path::String = ".")
    # Construct file names
    file1 = joinpath(path, "data_opt_$(casename).csv")
    file2 = joinpath(path, "data_opt_$(casename)_filtered.csv")
    file3 = joinpath(path, "data_opt_$(casename)_smooth_filtered.csv")

    # Read files
    df1 = CSV.read(file1, DataFrame)
    df2 = CSV.read(file2, DataFrame)
    df3 = CSV.read(file3, DataFrame)

    # Drop ρ_τ before renaming
    select!(df2, Not("ρ_τ"))
    select!(df3, Not("ρ_τ"))

    # Drop VinitFO before renaming
    select!(df2, Not("VinitFO"))
    select!(df3, Not("VinitFO"))    
    select!(df3, Not("VinitSO"))



    # Drop VinitFO before renaming
    select!(df2, Not("Vss"))
    select!(df3, Not("Vss"))



    # Rename columns (preserving :τb)
    rename!(df2, [Symbol(name) => Symbol("$(name)_filtered") for name in names(df2) if Symbol(name) != :τb])
    rename!(df3, [Symbol(name) => Symbol("$(name)") for name in names(df3) if Symbol(name) != :τb])

    # Join on :τb
    df_combined = innerjoin(df1, df2, on = :τb)
    df_combined = innerjoin(df_combined, df3, on = :τb)

    return df_combined
end


function summarize_optimal_taub_all(cases::Vector{Tuple{String, String}})
    # Columns to optimize
    value_cols = [:Vss, :VinitFO, :VinitSO, :VinitSO_filtered, :VinitSO_smooth]

    # Initialize empty DataFrame with proper columns
    summary = DataFrame(case = String[], 
                        Vss = Float64[], 
                        VinitFO = Float64[], 
                        VinitSO = Float64[], 
                        VinitSO_filtered = Float64[], 
                        VinitSO_smooth = Float64[])

    for (casename, filepath) in cases
        df = CSV.read(filepath, DataFrame)

        # Extract best τb for each metric
        best_taubs = [df.τb[argmax(df[!, col])] for col in value_cols]

        # Append to summary
        push!(summary, vcat([casename], best_taubs))
    end

    return summary
end
casenames = ["base", "lowchi", "highchi", "NoCol","sameinctax", "noinctax", "tauw10","tauw20","tauw30","tauw60"]
cases = Vector{Tuple{String, String}}()

for casename in casenames
    combined_df = read_and_merge_opt_data(casename)
    filename = "data_opt_combined_$(casename).csv"
    CSV.write(filename, combined_df)
    push!(cases, (casename, filename))
end

summary_df = summarize_optimal_taub_all(cases)
println(summary_df)
