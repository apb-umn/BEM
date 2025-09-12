# calibrated noise

include("OCModelEGMInputs.jl")
include("OCModelEGM.jl")
momfilename="lowsigma_moments.tex"


OCM=OCModel()
#OCM.χ = 3.0
OCM.σ_ε = 0.001
OCM.Na =200
OCM.Θ̄ =0.645

OCM.Nit = 500

rguess,trguess= 0.038695759441724556, 0.5292017623156571
OCM.rlb=rguess*.8
OCM.rub=rguess*1.2
OCM.trlb =trguess*.8
OCM.trub =trguess*1.2
OCM.r=rguess
OCM.tr=trguess
setup!(OCM)

OCM.ibise = 1

ichk = 0

if ichk==1
    chk1,chk2=check!(OCM)
else
    ss,lev,shr,res = solvess!(OCM)
    updatecutoffs!(OCM)
    moments=getMoments(OCM,savepath=momfilename)



    if OCM.iagg==1

        println("")
        println("                      US NIPA, 2007")
        println("                      =============")
        println("")
        println("      Incomes                   Products               ")
        println("    ---------------------------------------------------")
        println("      Sweat           1,491  |  Consumption    11,067  ")
        println("      Compensation    7,665  |    Private       9,548  ")
        println("        Nonbus        2,168  |    Nondefense    1,520  ")
        println("        C-Corp        3,665  |  Investment      4,847  ")
        println("        Pass-thru     1,832  |  Defense           679  ")
        println("      Capital income  7,437  |                         ")
        println("        Nonbus        2,233  |                         ")
        println("        Business      5,204  |                         ")
        println("    ---------------------------------------------------")
        println("      Adj GDP        16,593                    16,593  ")
        println("")
        println("")
        println("      Incomes (%GDP)            Products               ")
        println("    ---------------------------------------------------")
        println("      Sweat             9.0  |  Consumption      66.7  ")
        println("      Compensation     46.2  |  Investment       29.2  ")
        println("        Nonbus         13.1  |  Defense           4.1  ")
        println("        C-Corp         22.1  |                         ")
        println("        Pass-thru      11.0  |                         ")
        println("      Capital income   44.8  |                         ")
        println("    ---------------------------------------------------")
        println("      Adj GDP         100.0                     100.0  ")
        println("")
        println("")
        println("      Tax receipts              Expenditures           ")
        println("    ---------------------------------------------------")
        println("      Personal tax    1,492  |  Transfers       3,273  ")
        println("      I&P tax         1,037  |    NIPA          1,753  ")
        println("      Corporate tax     386  |    Nondefense    1,520  ")
        println("      Tax from ROW       14  |  Defense           679  ")
        println("                             |  Net interest      412  ")
        println("                             | *Deficit        -1,435  ")
        println("    ---------------------------------------------------")
        println("      Total           2,929                     2,929  ")
        println("")
        println("")
        println("      Tax receipts (%GDP)       Expenditures (%GDP)    ")
        println("    ---------------------------------------------------")
        println("      Personal tax      9.0  |  Transfers        19.7  ")
        println("      I&P tax           6.2  |    NIPA           10.6  ")
        println("      Corporate tax     2.3  |    Nondefense      9.2  ")
        println("      Tax from ROW      0.0  |  Defense           4.1  ")
        println("                             |  Net interest      2.5  ")
        println("                             | *Deficit          -8.6  ")
        println("    ---------------------------------------------------")
        println("      Total            17.7                      17.7  ")
        println("")
        println("      *includes SI, transfers, net saving              ")
        println("")
        println("")
        println("                    OC Model Results")
        println("                    ================")
        println("")
        println("      Equilibrium values and residuals")
        println("    -------------------------------------------")
        @printf("      Interest rate           %6.2f%%\n",ss[1])
        @printf("      Governemnt transfer     %6.2f\n",ss[2])
        @printf("      Asset market residual   %10.2e\n",ss[3])
        @printf("      Government budget       %10.2e\n",ss[4])
        println("")
        println("")
        println("      Incomes (%GDP)            Products             ")
        println("    -------------------------------------------------")
        @printf("      Sweat          %6.1f  |  Consumption    %6.1f\n",shr[6],shr[1])
        @printf("      Compensation   %6.1f  |  Investment     %6.1f\n",shr[7],shr[2]+shr[3])
        @printf("      Capital income %6.1f  |  Defense        %6.1f\n",100-shr[6]-shr[7],shr[4])
        println("    -------------------------------------------------")
        println("      Adj GDP         100.0                     100.0  ")
        println("")
        println("")
        println("      Tax receipts (%GDP)      Expenditures          ")
        println("    -------------------------------------------------")
        @printf("      Sweat income   %6.1f  |  Transfers      %6.1f\n",shr[12],shr[20])
        @printf("      Employee wages %6.1f  |  Defense        %6.1f\n",shr[13],shr[18])
        @printf("      Profits        %6.1f  |  Net interest   %6.1f\n",shr[14],shr[19])
        @printf("      Dividends      %6.1f  |                      \n",shr[15])
        @printf("      Consumption    %6.1f  |                      \n",shr[16])
        println("    -------------------------------------------------")
        @printf("      Total          %6.1f                    %6.1f\n",
                       shr[12]+shr[13]+shr[14]+shr[15]+shr[16],shr[18]+shr[19]+shr[20])
    end 
end


# Alternative version that returns only the aggregate labor demand
function aggregate_labor_demand(OCM)
    """
    Calculate only the aggregate labor demand (more efficient if intermediates not needed).
    
    Parameters:
    - OCM: Object containing model parameters and functions
    
    Returns:
    - Nb: Aggregate labor demand (scalar)
    """
    
    ah = OCM.alθ[1:OCM.Ia, 1]
    nb = hcat([OCM.bf.n[s](ah) for s in 1:OCM.Nθ]...)
    nbdst = [zeros(OCM.Ia * OCM.Nθ); nb[:]]
    Nb = dot(OCM.ω, nbdst)
    
    return Nb
end

τ̂b = 0.05
function get_p(OCMhat)
    @unpack Vcoefs,wf,bf,Nθ,lθ,πθ,Ia,alθ,r,w,σ_ε = OCMhat

    ah  = alθ[1:Ia,1] #grids are all the same for all shocks
    Vw  = hcat([wf.v[s](ah) for s in 1:Nθ]...)
    Vb  = hcat([bf.v[s](ah) for s in 1:Nθ]...)
    p̂    = 1.0 .- probw.(Vb.-Vw,σ_ε)
    return p̂
end


pvec = get_p(OCM)
OCM′ = deepcopy(OCM)
OCM′.τb += τ̂b 
#OCM′.τw += τ̂b
solve_eg!(OCM′)
dist!(OCM′)

se_rate = sum(reshape(OCM.ω,:,2)[:,2])
se_rate′ = sum(reshape(OCM′.ω,:,2)[:,2])

ah  = OCM.alθ[1:OCM.Ia,1] #grids are all the same for all shocks
nb     = hcat([OCM.bf.n[s](ah) for s in 1:OCM.Nθ]...) #labor demand from business
nbdst  = [zeros(OCM.Ia*OCM.Nθ);nb[:]]
Nb     = dot(OCM.ω,nbdst) #agg labor demand from business

Nb=aggregate_labor_demand(OCM)
Nb′=aggregate_labor_demand(OCM′)
elasticityNB= log(Nb′/Nb)/log((1-OCM′.τb)/(1-OCM.τb))
semi_elasticityNB = (Nb′ - Nb) / Nb / (OCM′.τb - OCM.τb)

println("ssemi_elasticityNB: ", semi_elasticityNB)


# sweat income share of 11%
# compensation to workers/Y of 45 percent
## WNb/Y = 0.11%
## WNc/Y = 0.34%
# business loan-to-GDP ratio = 13%
# tax revenues/Y of  26%
# net interest/Y of 2.5%
# fraction of owners 25%