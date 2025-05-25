# OCDriverE.jl
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
#
#   Workers:
#
#         vw(a,θw,θb) =  max U(c) + β E v(a',θw',θb',η')
#                       c,a'
#
#           s.t. c+a' = (1+r)*a + w*θw -τw*w*θw -τc*c
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

#   Ellen McGrattan, 5/9/2025
#   Revised, ERM, 5/25/2025


include("OCModelE.jl")
OCM=OCModel()
setup!(OCM)
ichk = 0

if ichk==1
    chk1,chk2=check!(OCM)
else
    ss,lev,shr = solvess!(OCM)


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
