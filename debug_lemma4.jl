
include("OCModelEGMInputs.jl")
include("OCModelEGM.jl")
include("OCModelEGM_transition.jl") # has the F,G,ff and several helper functions

# Define parameters
τb_val = 0.40
ρ_τ_val_fast = 0.0
ρ_τ_val_slow = 0.9

# Create output filenames using interpolation
filenamefast = "df_transition_fast_$(round(τb_val, digits=2)).csv"
filenameslow = "df_transition_slow_$(round(τb_val, digits=2)).csv"
saveplotfilename = "transition_comparison_$(round(τb_val, digits=2)).pdf"


τb_val, ρ_τ_val_fast, ρ_τ_val_slow, filenamefast, filenameslow, saveplotfilename =     τb_val, ρ_τ_val_fast, ρ_τ_val_slow, filenamefast, filenameslow, saveplotfilename
println("Setting up old steady state (takes a few minutes)...")
OCM_old = OCModel()
setup!(OCM_old)
OCM_old.r = 0.03867770200367673
OCM_old.tr = 0.46806124541903205
_, X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old = setup_old_steady_state!(OCM_old)
println("Old steady state setup complete.")

println("Setting up new steady state with τb = $τb_val (takes a few minutes)...")
OCM_new, _ = setup_new_steady_state(τb_val, OCM_old.τw, OCM_old)
println("New steady state setup complete.")

# --- FAST TRANSITION ---
OCM_new.ρ_τ = ρ_τ_val_fast
println("Performing transition analysis with ρ_τ = $ρ_τ_val_fast...")

X̄_0,Ix̄_0,A_0, Taub_0, ω̄_0_base, OCM_new=X̄_old, Ix̄_old, A_old, Taub_old, ω̄_0_old, OCM_new

println("→ Constructing inputs...")
inputs = construct_inputs(OCM_new)
println("...done")

println("→ Zeroth-order approximation...")
ZO = ZerothOrderApproximation(inputs)
println("...done")

println("→ Computing derivatives...")
computeDerivativesF!(ZO, inputs)
computeDerivativesG!(ZO, inputs)
println("...done")

println("→ Setting up first-order approximation object...")
FO = FirstOrderApproximation(ZO, OCM_new.T)
println("...done")

println("→ Computing x,M,L,Js components ( bulk of the calclations )...")
compute_f_matrices!(FO)
compute_Lemma3!(FO)


"""
    compute_Lemma4!(FO)

Computes the terms from Lemma 4, Operators L and terms a_s = M p x_s 
"""
function compute_Lemma4!(FO)
    @unpack ZO,x,T = FO
    @unpack Φ,Φₐ,Δ,p,pκ,ω̄,Λ,x̄,n,dlΓ = ZO
    #Λ is other the full joint distribution
    #Γ and dlΓ are vectors.
    #(x̄*Δ) computes the coefficients for x̄Δ= x̄⁻ - x̄⁺
    #compute Γκ̄
    
    #start with computing La 
    #ā_a = reshape((p*x̄)*Φₐ,n.a,:)  #ā_a is n.a x (n.a x n.Ω )(compress last two dimensions)
    x̄_a = reshape(FO.x̄_a,n.x,n.a,n.sp)
    ā_a =  reshape(p*reshape(x̄_a,n.x,n.a,:)*Φ,n.a,:)
    FO.La = La = kron(Λ,ones(n.a,n.a)) #initialize La operator 

    rowsLa = rowvals(La)
    for j in eachindex(ā_a[1,:]) #iterate over all columns
        for index in nzrange(La,j)
            i = rowsLa[index]
            ia = (i-1)%n.a+1
            @inbounds La.nzval[index] *= ā_a[ia,j]
        end
    end

    #Next the L operator
    κ̄_a = reshape((pκ*x̄_a)*Δ*Φ,n.a,:)

    
    dlΓκ̄_a = (dlΓ'.*reshape(κ̄_a,n.a,:))[:]
    FO.L = kron(Λ,ones(1,n.a)).*dlΓκ̄_a' #initialize L operator

    #Next compute a objects
    #Iz = length(ω̄)
    Ina = Matrix(I,n.a,n.a)
    ΛIna = kron(Λ,Ina)
    #Ma is now (n.a x n.Ω) x (n.a x n.sp) matrix
    FO.Ma = ΛIna*kron(Φ'.*ω̄,Ina)
    FO.M =  Λ*(dlΓ.*ω̄.*Φ')
    
    
    FO.a = a = zeros(n.sp*n.a,n.Q,T)
    FO.κ = κ = zeros(n.sp,n.Q,T)
    for s in 1:T
        #x[s] is n.x x n.Q x n.sp
        as = permutedims(p*x[s],[1,3,2]) #n.a x n.sp x n.Q
        @views a[:,:,s] .= reshape(as,:,n.Q)
        κs = permutedims(pκ*x[s]*Δ,[1,3,2]) #n.sp x n.Q
        @views κ[:,:,s] .= reshape(κs,:,n.Q)
    end

    #and now the I operators
    FO.I  = x̄*Φ #aggregate changes in density
    dlΓκ̄_a_mat = kron(sparse(I,n.Ω,n.Ω),ones(1,n.a)).*dlΓκ̄_a'
 
    FO.Ia = reshape(x̄_a*Φ,n.x*n.a,n.Ω) .+ (x̄*Φ)*dlΓκ̄_a_mat   #aggregate changes in state 
    #FO.Ia = combine_terms(x̄_a, x̄, Φ, dlΓκ̄_a_mat)  #aggregate changes in state 
    GC.gc()
end

compute_Lemma4!(FO)
