# OCModelEGMInputs.jl
#
#   Three business sectors:
#     -- Nonfinancial corporate sector:
#
#         vc(Kc) =  max {(1-τd)dc +(1+γ)/(1+r) vc(Kc')}
#                  Nc,Xc 
#
#           s.t. (1+γ)Kc'=(1-δ)Kc+Xc
#                     Yc = Θ̄ Kc^α Nc^(1-α)
#                     dc = Yc-w*Nc-Xc-τp(Yc-w*Nc-δ*Kc)
#
#    -- Financial corporate sector
#         vi(x) =  max  {di+ (1+γ)/(1+r) vi(x')} 
#                   x'    
#
#           s.t. di = q*s+b+∫kb-∫a 
#                      +(1-τd)dc*s+r*b-rp∫kb-r∫a (income)
#                      -(1+γ)[q*s'+b'-∫a']-∫[kb'-(1-δ)kb]
#                   = net worth+income-invetment
#
#        ==> r=rp-δ=(1-τd)*[q+(1-τd)dc]/q[-1]-1
#                =(1-τp)(α*Yc/Kc-δ)
#
#    -- Private sector operating DRS technology 
#
#         vb(a,θw,θb) =  max U(c) + β E v(a',θw',θb',η')
#                        c,a'
#
#           s.t. c+a' = (1+r)a + π - τb*π-τc*c
#                π = max θb kb^α nb^ν-(r+δ)*kb-w*nb
#                   a'≥ 0
#                   k≤ χ*a + k_min
#
#   Workers:
#
#         vw(a,θw,θb) =  max U(c) + β E v(a',θw',θb',η')
#                       c,a'
#
#           s.t. c+a' = (1+r)*a + w*θw -τw*w*θw -τc*c
#                a'≥ 0
#
#   Occupation choice:
#
#         η = taste shock = ηw-ηb ~ logistic
#         p = probability of being a worker
#         v = max d*(vw+η) + (1-d)*vb
#
#        Ev = 0.57722*σ_η + σ_η *log(exp(vw/σ_η)+exp(vb/σ_η))
#           = 0.57722*σ_η + vw+ σ_η*(1+exp((vb-vw)/σ_η))
#
#   Market clearing:
#
#        Kc + ∫ (1-d[i])k[i] di = ∫ a[i] di
#        Nc + ∫ (1-d[i])n[i] di = ∫ d[i] θw[i] di
#
#   Government budget:
#
#        G+Tr+r*B =B'-B+τc*C+τw*w*(Nc+Nb)
#                  +τp*(Yc-w*Nc-δ*Kc)
#                  +τd*(Yc-w*Nc-Xc-τp*(Yc-w*Nc-δ*Kc))
#                  +τb*(Yp-(r+δ)*Kp-w*Nb)
#

using Parameters,LinearAlgebra,BasisMatrices,SparseArrays,Arpack,Roots, 
      KrylovKit,QuantEcon,StatsBase,ForwardDiff,Dierckx,
      Plots,NPZ,NLsolve,Printf,DataFrames,CSV,Distances, Interpolations
      
rguess,trguess=0.05923207359775146, 0.6401956237429345

"""
Parameters of the Occupation Choice Model
"""
@with_kw mutable struct OCModel

    #Preferences
    σ::Float64   = 1.5                  #Risk Aversion
    βo::Float64  = 0.97                 #Discount Factor (original)
    γ::Float64   = 0.02                 #Economy growth rate
    βEE::Float64 =  βo*(1+γ)^(-σ)           #Discount Factor for EE (with growth) 
    βV::Float64 = βo*(1+γ)^(1-σ)           #Discount Factor for Value function (with growth) 
    σ_ε::Float64 = 0.4                 #St.Dev. of taste shock ε

    #Corporate parameters
    α::Float64   = 0.45                  #Corporate capital share
    Θ̄::Float64   = 1.15                #Corporate TFP
    δ::Float64   = 0.041                #Depreciation rate

    #Entrepreneur parameters
    α_b::Float64 = 0.33                 #Private capital share
    ν::Float64   = 0.33                 #Private labor share 
    χ::Float64   = 1.25                  #Collateral constraint. Use 1.5 for comparative stats
    k_min::Float64 = 1e-2                #Minimum capital (not subject to collateral constraint)


    #Entrepreneur income shocks
    N_θb::Int     = 5                   #Number of productivity shocks θb
    ρ_θb::Float64 = 0.966               #Persistence of θb
    σ_θb::Float64 = 0.20                #St. Dev. of θb
    bm_θb::Int    = 1                   #Use estimates from BM (2021)
    πθbBM::Matrix{Float64} = [
    0.611519   0.170401    0.0983162  0.0645004  0.0552636;
    0.172256   0.550903    0.187292   0.0643231  0.0252256;
    0.0986772  0.19074     0.475423   0.190404   0.0447556;
    0.0599203  0.0546977   0.1637     0.558095   0.163587;
    0.0455165  0.00943903  0.0342529  0.135187   0.775604
    ]
    θbgridBM::Vector{Float64} = [
        0.4316379803912925,
        0.656991613638479,
        1.0,
        1.52208944412838,
        2.316756275927041
    ]

    #Worker income shocks 
    N_θw::Int     = 5                   #Number of productivity shocks θw
    ρ_θw::Float64 = 0.966               #Persistence of θw
    σ_θw::Float64 = 0.13                #St.Dev. of θw
    bm_θw::Int    = 1                   #Use estimates from BM (2021)see below
    πθwBM::Matrix{Float64} = [
    0.423678      0.548952      0.0273418   2.73722e-5   3.92763e-10;
    0.0463499     0.620821      0.327374    0.00545366   1.58881e-6;
    0.000761779   0.144558      0.70936     0.144558     0.000761779;
    1.58881e-6    0.00545366    0.327374    0.620821     0.0463499;
    3.92763e-10   2.73722e-5    0.0273418   0.548952     0.423678   
    ]
    θwgridBM::Vector{Float64} = [
    0.5088696869303454,
    0.7133510264451474,
    1.0,
    1.4018343885804927,
    1.9651396530068437
    ]
    risk_adjust::Float64 = 1.0         #Risk adjustment for worker productivity shocks

    #Asset grids
    a̲::Float64    = 0                 #Borrowing constraint
    amax::Float64 = 200.             #Maximum asset grid point
    Na::Int       = 100                  #Number of gridpoints for splines
    so::Int       = 2                   #Spline order for asset grid
    Ia::Int       = 1000                #Number of gridpoints for histogram
    curv_a::Float64 = 2.0               #Controls spacing for asset grid
    curv_h::Float64 = 3.0               #Controls spacing for histogram

    #Fiscal policy
    g::Float64    = 0.11                #Government spending on G&S
    b::Float64    = 3.0                 #Debt
    τb::Float64   = 0.20                 #Tax on private business
    τw::Float64   = 0.37                 #Tax on wages
    τp::Float64   = .20               #Tax on corporate profits
    τd::Float64   = 0.0               #Tax on dividend
    τc::Float64   = 0.06               #Tax on consumption
    tx::Float64   = 0.0                 #Total tax
    ρ_τ::Float64  = 0.9
    
    #Numerical parameters
    Nhoward::Int = 1               #Number of iterations in Howard's method
    trlb::Float64 = trguess*0.8                 #Transfers lower bound
    trub::Float64 = trguess*1.2                 #Transfers upper bound
    Ntr::Int      = 1                   #Number of transfer evaluations
    rlb::Float64  = rguess*.8
    #Rate lower bound 
    rub::Float64  =rguess*1.2              #Rate upper bound 
    Nr::Int       = 3                   #Number of rate evaluations (in check!)
    Neval::Int    = 2                   #Number of bisection evaluations
    iagg::Int     = 1                   #Show aggregate data for each r/tr combo
    λ::Float64    = 1.0                 #Weight on Vcoefs update
    Nit::Int      = 500                #Number of iterations in solve_eg!
    tolegm::Float64 = 1e-6            #Tolerance for EGM convergence
    ftolmk::Float64 = 1e-4            #Tolerance for market clearing
    xtolxmk::Float64 = 1e-4            #Tolerance for market clearing
    maxitermk::Int = 500         #Maximum iterations for market clearing
    ibise::Int = 1              #Bisection method for initial guess of r/tr
    iprint::Int   = 1                   #Turn on=1/off=0 intermediate printing
    T::Int        = 500                 #Number of periods in solve_tr!
    ξ::Float64    = 1.                 #Newton relaxation parameter
    inewt::Int    = 1                   #Use simple Newton in solvess
    diffv::Float64 = 1.              #Difference for egm 

    #Vector/matrix lengths
    Nθ::Int       = N_θb*N_θw           #Total number of income shocks
    Nv::Int       = Nθ*Na               #Length of V coefficients
    Nh::Int       = Nθ*Ia*2             #Total number of histogram states

    #
    # Values to be filled in setup!
    #   agrid = grid vector for endogenous grid
    #   lθ = log productivity shocks [log(θb),log(θw)]
    #   πθ = transition matrix for θ
    #   Φ = basis matrix, ie, Φ*f = f(a,θ), for all a,θ
    #   EΦeg = expectation on gridpoints next period, eg EΦeg*fcoefs 
    #        = ∑ πθ(s,s′) ∑ fcoeff^j ϕ^j(a′,s′) 
    #   EΦ_aeg = derivative of the expectation with respect to assets
    #

    agrid::Vector{Float64} = zeros(Na)  
    abasis::Basis{1,Tuple{SplineParams{Vector{Float64}}}} =Basis(SplineParams(collect(LinRange(0,1,Na-1)),0,2))
    πθ::Matrix{Float64} =zeros(Nθ,Nθ)   
    lθ::Matrix{Float64} = zeros(Nθ,2)    
    Φ::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na)
    EΦeg::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na)  
    EΦ_aeg::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na) 

    #
    # Equilibrium results to be initialized
    #   alθ = gridpoints for the stationary distribution
    #   ω = stationary distribution 
    #   Λ = transition matrix for the stationary distribution
    #   r = interest rate
    #   w = wage rate
    #   tr = government transfers
    #   Vcoefs = coefficients of value function
    #

    alθ::Matrix{Float64} = zeros(Nθ*Ia,3) 
    ω::Vector{Float64} = zeros(Nh)   
    Λ::SparseMatrixCSC{Float64,Int64} = spzeros(Ia,Ia)
    r::Float64  = rguess  
    w::Float64  = 2.0
    tr::Float64 =trguess
    Vcoefs::Vector{Float64} = zeros(Nv) 
    λcoefs::Vector{Float64} = zeros(Nv)
    wf::NamedTuple = (c=Vector{Spline1D}(), a=Vector{Spline1D}(), v=Vector{Spline1D}())
    bf::NamedTuple = (c=Vector{Spline1D}(), a=Vector{Spline1D}(), v=Vector{Spline1D}(),
                      k=Vector{Spline1D}(), n=Vector{Spline1D}(), y=Vector{Spline1D}(), π=Vector{Spline1D}())
    egi::Vector{Spline1D}=Vector{Spline1D}(undef,Nθ)
    ab_col_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds
    ab_bor_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds
    aw_bor_cutoff::Dict{Vector{Float64},Float64} = Dict{Vector{Float64},Float64}() #Stores the points at which the borrowing constraint binds


end


"""
    export_parameters_for_paper(model::OCModel, filepath::String) -> DataFrame

Export a summary of selected model parameters to a CSV file for use in a paper or report.

# Arguments
- `model::OCModel`: An instance of the occupation choice model containing calibrated parameters.
- `filepath::String`: The file path (including `.csv` extension) where the parameter table will be saved.

# Returns
- `DataFrame`: A table with columns: `Row`, `Description`, `Expression`, and `Value`.

# Notes
- The function maps model parameters to paper-friendly notation and labels.
- The row numbering is generated automatically based on the number of entries.
- Designed for consistency between code and documentation output.

# Example
```julia
model = OCModel()
export_parameters_for_paper(model, "parameters_table.csv")
"""

function export_parameters_for_paper(model::OCModel, filepath::String)

    # Descriptions, expressions, and values
    descriptions = [
        "RelRiskAver", "DiscFactor", "Growth",
        "PBcapshr", "PBlabshr", "CCTFP", "CCcapshr", "CCdepr",
        "Alowerbnd", "Maxleverage", "Defspend", "Debt",
        "TaxWage", "TaxPBinc", "TaxCCinc", "TaxCons", "RhoTau"
    ]

    expressions = [
        "mu", "beta", "gamma",
        "phi", "nu", "Theta", "alpha", "delta",
        "amin", "chiminus", "G", "B",
        "tauw", "taub", "taup", "tauc", "rhotau"
    ]

    values = [
        model.σ, model.βo, model.γ,
        model.α_b, model.ν, model.Θ̄, model.α, model.δ,
        model.a̲, model.χ - 1.0, model.g, model.b,
        model.τw, model.τb, model.τp, model.τc, model.ρ_τ
    ]

    nrows = length(values)

    params_table = DataFrame(
        Row = 1:nrows,
        Description = descriptions,
        Expression = expressions,
        Value = values
    )

    CSV.write(filepath, params_table)
    return params_table
end




function export_BM_tables(model::OCModel, filepath::String)

    # Format number with required precision and style
    format_num(x) = round(x, digits=3) ≈ 0 ? "0" :
        replace(@sprintf("%.3f", x), r"^0(?=\.)" => "") |> 
        x -> replace(x, r"0+$" => "") |> 
        x -> replace(x, r"\.$" => "")

    # Format a row of floats into a CSV-formatted string with 6 columns
    function format_row(row::Vector{Float64})
        s = [format_num(x) for x in row]
        padded = vcat(s, fill("", 6 - length(s)))
        return join(padded, ",")
    end

    # Format matrix rows
    function format_matrix(mat::Matrix{Float64})
        return [format_row(collect(row)) for row in eachrow(mat)]
    end

    # Prepare all rows
    row1 = format_row(model.θbgridBM)
    rows_b = format_matrix(model.πθbBM)
    row2 = format_row(model.θwgridBM)
    rows_w = format_matrix(model.πθwBM)

    # Combine everything
    data_rows = vcat([row1], rows_b, [row2], rows_w)

    # Open file and write all rows manually
    open(filepath, "w") do io
        println(io, "Row,Column1,Column2,Column3,Column4,Column5,Column6")
        for (i, row) in enumerate(data_rows)
            println(io, string(i) * "," * row)
        end
    end

    return nothing
end
