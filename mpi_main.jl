#!/usr/bin/env julia
using MPI
using Random
using Printf
using Dates
using JLD2
using Statistics
using LinearAlgebra
#######################################################
# Pauli matrices module
#######################################################
module Pauli
export σx, σy, σz

const σx = [0 1; 1 0]
const σy = [0 -im; im 0]
const σz = [1 0; 0 -1]

end 
#######################################################

using LinearAlgebra
using Random
using Printf
using Statistics
using JLD2
using .Pauli


const S_MAG = 1.5            # classical spin magnitude

#######################################################
# Random spin + spin rotation
#######################################################
function rotate_spin_angles(θ, φ)
    max_angleθ=0.05
    max_angleφ =0.1
    
    θ_new = θ + (rand()-0.5)*max_angleθ
    φ_new = φ + (rand()-0.5)*max_angleφ 
   

    return θ_new, φ_new
end


function random_spin_angles()
    θ = acos(2rand() - 1)      # θ ∈ [0, π]  
    φ = 2π*rand()              # φ ∈ [0, 2π]
    return (θ, φ)
end

# convert angles → 3D spin
function angles_to_vector(θ, φ)
    S_MAG * [sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
end




function build_neighbors(Lx, Ly)

    idx(x, y) = x + (y-1)*Ly
    index_to_xy(i) = (i - (fld(i-1, Ly))*Ly, fld(i-1, Ly) + 1)


    N = Lx * Ly
    neigh    = [Int[] for _ in 1:N]
    neighNNN = [Int[] for _ in 1:N]

    for y in 1:Ly, x in 1:Lx
        i = idx(x, y)

        xp = mod1(x+1, Lx)
        xm = mod1(x-1, Lx)
        yp = mod1(y+1, Ly)
        ym = mod1(y-1, Ly)
     
        push!(neigh[i], idx(xp, y))
        push!(neigh[i], idx(xm, y))
        push!(neigh[i], idx(x, yp))
        push!(neigh[i], idx(x, ym))



        if  isodd(x) && isodd(y)

            push!(neighNNN[i], idx(xp, yp))   # (x+1, y-1)
            push!(neighNNN[i], idx(xm, ym))   # (x-1, y+1)


            # println("  Site $i ($(x),$(y))         NNN →  ", [index_to_xy(n) for n in neighNNN[i]])
            # println()


        end

        if  iseven(x) && isodd(y)


            push!(neighNNN[i], idx(xp, ym))   # (x+1, y-1)
            push!(neighNNN[i], idx(xm,yp))   # (x-1, y+1)


            # println("  Site $i ($(x),$(y))         NNN →  ", [index_to_xy(n) for n in neighNNN[i]])
            # println()
        end

        if  iseven(y) && isodd(x)

            push!(neighNNN[i], idx(xp, ym))   # (x+1, y-1)
            push!(neighNNN[i], idx(xm, yp))   # (x-1, y+1)


            # println("  Site $i ($(x),$(y))         NNN →  ", [index_to_xy(n) for n in neighNNN[i]])
            # println()
        end


        if  iseven(y) && iseven(x)

            push!(neighNNN[i], idx(xp, yp))   # (x+1, y-1)
            push!(neighNNN[i], idx(xm, ym))   # (x-1, y+1)


            # println("  Site $i ($(x),$(y))         NNN →  ", [index_to_xy(n) for n in neighNNN[i]])
            # println()
        end




    end

    return neigh, neighNNN
end



function build_hamiltonian(S_angles, neigh, neighNNN;t1=t1,t2=t2, JH=JH, Jafm=Jafm)
    N = length(S_angles)

    H = zeros(ComplexF64, 2N, 2N)      
    E_classical=0.00
    E_aniso=0.0

    S = [angles_to_vector(θ, φ) for (θ, φ) in S_angles]

    # println(t1,t2,JH,Jafm)
    for i in 1:N
        for j in neigh[i]
      
            if j > i 
                H[2i-1, 2j-1] = -t1
                H[2j-1, 2i-1] = -t1

                H[2i,   2j]   = -t1
                H[2j,   2i]   = -t1

                E_classical += Jafm * dot(S[i], S[j])
        
            end

       
        end
    end
    
    for i in 1:N
        _, _, S1z = S[i]
    	E_aniso += 0.05 * (S1z^2)
    end

    for i in 1:N
        for j in neighNNN[i]

            if j > i 
                # println(i," ",j)
                H[2i-1, 2j-1] = -t2
                H[2j-1, 2i-1] = -t2
                
                H[2i,   2j]   = -t2
                H[2j,   2i]   = -t2
            end
        end
    end


    for i in 1:N
        Sx, Sy, Sz = S[i]
        M = Sx*Pauli.σx + Sy*Pauli.σy + Sz*Pauli.σz
        H[2i-1:2i, 2i-1:2i] = (-JH/2.0) * M
        # println(M)
    end
    # error("Stopping here")

    return Hermitian(H),E_classical+E_aniso
end


function find_mu_bisection(evals::Vector{Float64}, fill::Float64, T::Float64)
    dim_1 = length(evals)

    # Right bound: mR = maxval(evl_s)
    mR = maximum(evals)
    fR = 0.0
    for i in 1:dim_1
        fR += 1.0 / (exp((evals[i] - mR) / T) + 1.0)
    end

    # Left bound: mL = minval(evl_s)
    mL = minimum(evals)
    fL2 = 0.0
    for i in 1:dim_1
        fL2 += 1.0 / (exp((evals[i] - mL) / T) + 1.0)
    end

    # Middle point
    m_d = 0.5 * (mL + mR)
    f = 0.0
    for i in 1:dim_1
        f += 1.0 / (exp((evals[i] - m_d) / T) + 1.0)
    end

    # Bisection loop
    while abs(f - fill) >= 1e-8
        m_d = 0.5 * (mL + mR)

        # println(f," ",fill)

        f = 0.0
        for i in 1:dim_1
            f += 1.0 / (exp((evals[i] - m_d) / T) + 1.0)
        end

        if f > fill
            # filling too high → reduce chemical potential
            mR = m_d
            fR = f
        elseif f < fill
            # filling too low → increase chemical potential
            mL = m_d
            fR = f
        end
    end

    return m_d
end



fermi(E, mu, T) = 1.0/(1.0 + exp((E - mu) / T))

function electronic_energy_T(evals::Vector{Float64}, mu::Float64, T::Float64)
    E = 0.0
    @inbounds for e in evals
        E += e * fermi(e, mu, T)
    end
    return E
end


# -----------------------------
# Monte Carlo at a single temperature
# -----------------------------


function run_mc_temperature(T;
            Lx=0, Ly=0,
            nelec_frac=0.5,
            t1=0.0,
            t2=0.0,
            JH=0.0, Jafm=0.0,
            Nsteps=4000, Ntherm=2000, meas_interval=10,
            max_angle=0.3,neigh=neigh, neighNNN=neighNNN,
            initial_angles=nothing)      #initial_angles initiliaze later

    # -------------------------
    # Initialization
    # -------------------------
    Nsites = Lx*Ly
    

    filling = nelec_frac*2.0*Nsites

    # spin angles (θ, φ) per site
    S_angles = initial_angles === nothing ?
               [random_spin_angles() for _ in 1:Nsites] :
               deepcopy(initial_angles)

    β = 1.0 / T

    # -------------------------
    # Measurement arrays
    # -------------------------
    measurements_energy=Float64[]
    measurements_mx=Float64[]
    measurements_my=Float64[]
    measurements_mz=Float64[]
    measurements_mAx=Float64[]
    measurements_mAy=Float64[]
    measurements_mAz=Float64[]
    measurements_accept=Float64[]
    spin_snapshots= Vector{Vector{Vector{Float64}}}()   #spin_snapshots[T_step][site_index] = [Sx, Sy, Sz]

    # -------------------------
    # Staggered AFM sign factor η(i) = (-1)^(x+y)
    # -------------------------
    eta = [(-1)^(((i-1) ÷ Lx) + ((i-1) % Lx)) for i in 1:Nsites]
  
    # -------------------------
    # Initial Hamiltonian & energy
    # -------------------------

    H, E_classical = build_hamiltonian(S_angles, neigh, neighNNN; t1=t1,t2=t2, JH=JH, Jafm=Jafm)
    evals = real(eigvals(Hermitian(H)))

    # for (i, λ) in enumerate(evals)
    #     @printf("%3d  %.10f\n", i, λ)
    #     flush(stdout)
    # end
    

    mu= find_mu_bisection(evals, filling, T)

    E_elec= electronic_energy_T(evals,mu,T)
    E_total = E_elec + E_classical

    # -------------------------
    # Monte Carlo loop
    # -------------------------
    for step in 1:Nsteps
        n_accept = 0

        for i in 1:Nsites#randperm(Nsites)
            θ_old, φ_old = S_angles[i]
            θ_new, φ_new = rotate_spin_angles(θ_old, φ_old)
            S_angles[i] = (θ_new, φ_new)

            Hnew, E_classical_new = build_hamiltonian(S_angles, neigh, neighNNN; t1=t1,t2=t2, JH=JH, Jafm=Jafm)
            evals_new = real(eigvals(Hermitian(Hnew)))
            mu_new= find_mu_bisection(evals_new , filling, T)

            E_elec_new= electronic_energy_T(evals_new,mu_new, T)
            E_total_new = E_elec_new + E_classical_new
            
            ΔE = E_total_new - E_total
            # println(ΔE)
            if ΔE <= 0 || rand() < exp(-β * ΔE)
                E_total = E_total_new
                E_elec = E_elec_new
                n_accept += 1
            else
                S_angles[i] = (θ_old, φ_old)
            end
        end
        
        # -------------------------
        # Measurements   #first average on mx,my,mz then sqrt(<mx>**2+...+...)
        # -------------------------
        if step > Ntherm && step % meas_interval == 0
            push!(measurements_energy, E_total)

            # convert to spin vectors
            S_vectors = [angles_to_vector(θ, φ) for (θ, φ) in S_angles]


            mx = sum(s[1] for s in S_vectors) / Nsites
            my = sum(s[2] for s in S_vectors) / Nsites
            mz = sum(s[3] for s in S_vectors) / Nsites

            push!(measurements_mx, mx)
            push!(measurements_my, my)
            push!(measurements_mz, mz)


            # staggered AFM order


            mAx = sum(eta[i] * S_vectors[i][1] for i in 1:Nsites)/Nsites
            mAy = sum(eta[i] * S_vectors[i][2] for i in 1:Nsites)/Nsites
            mAz = sum(eta[i] * S_vectors[i][3] for i in 1:Nsites)/Nsites


            push!(measurements_mAx, mAx)
            push!(measurements_mAy, mAy)
            push!(measurements_mAz, mAz)
          
            push!(spin_snapshots, deepcopy(S_vectors))

            push!(measurements_accept, n_accept / Nsites)
        end
    end

    return (measurements_energy,measurements_mx, measurements_my, measurements_mz,measurements_mAx, measurements_mAy, 
          measurements_mAz,measurements_accept,spin_snapshots,S_angles)

    
end


# -----------------------------
# Run full simulation over temperatures
# -----------------------------
function run_full_simulation(JH, Jafm,t1,t2; Tmin, Tmax, NT,Lx,Ly)

    println(t1," ",t2," ",JH, " ", Jafm, " ", Tmin, " ", Tmax, " ", NT, " ", Lx, " ", Ly)
    flush(stdout)

    Temps = exp.(range(log(Tmax), log(Tmin), length=NT))

    results_energy = Dict{Float64, Vector{Float64}}()

    # store averaged uniform and AFM magnetizations
    results_m_avg  = Dict{Float64, Float64}()
    results_mAF_avg = Dict{Float64, Float64}()

    results_acc    = Dict{Float64, Vector{Float64}}()
    spin_configs   = Dict{Float64, Vector{Vector{Vector{Float64}}}}()

    last_spins = nothing
    neigh, neighNNN = build_neighbors(Lx, Ly)

    for (k, T) in enumerate(Temps)
        @printf("Running T = %.3f   (%d / %d)\n", T, k, length(Temps))
        flush(stdout)

    
        Elist,mx_list, my_list, mz_list,mAx_list, mAy_list, mAz_list,Alist,snapshots,
        last_spins =run_mc_temperature(T; Lx=Lx, Ly=Ly,initial_angles=last_spins,JH=JH, Jafm=Jafm,t1=t1,t2=t2,
                                        neigh=neigh, neighNNN=neighNNN)

  
        results_energy[T] = Elist
        results_acc[T]    = Alist
        spin_configs[T]   = snapshots

        # ------------------------------
        #  ⟨m⟩ and ⟨m_AF⟩ 
        # ------------------------------
        mx_avg  = mean(mx_list)
        my_avg  = mean(my_list)
        mz_avg  = mean(mz_list)

        mAx_avg = mean(mAx_list)
        mAy_avg = mean(mAy_list)
        mAz_avg = mean(mAz_list)

        m_avg   = sqrt(mx_avg^2 + my_avg^2 + mz_avg^2)
        mAF_avg = sqrt(mAx_avg^2 + mAy_avg^2 + mAz_avg^2)

        

        results_m_avg[T]  = m_avg
        results_mAF_avg[T] = mAF_avg
    end

    return Temps, results_energy, results_m_avg, results_mAF_avg, results_acc, spin_configs
end




function main()
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if rank == 0
        println("[Main] MPI started with $size ranks at ", Dates.now())
    end


    JH_arr = [0.0, 2.0, 4.0, 8.0]
    Jafm_arr = [0.0, 0.25, 0.5, 1.0]

    param_list = [(JH, Jafm) for JH in JH_arr, Jafm in Jafm_arr]
    param_list = collect(Iterators.flatten(param_list))

    my_params = [param_list[i] for i in rank+1:length(param_list):size]

    println("[rank $rank] assigned parameters: ", my_params)

    for (JH, Jafm) in my_params
        println("[rank $rank] Running simulation with JH=$JH, Jafm=$Jafm at ", Dates.now())
        Temps, results_energy, results_mag, results_mAF, results_acc, spin_configs =
            run_full_simulation(JH, Jafm, t1=1.0, t2=0.5; Tmin=0.001, Tmax=1.0, NT=30, Lx=8, Ly=8)

        # Save output files per rank
        fname_steps = @sprintf("MC_energy_mag_vs_T_and_step_JH%.3f_Jafm%.3f.txt", JH, Jafm, rank)
        fname_avg   = @sprintf("MC_avg_energy_mag_vs_T_JH%.3f_Jafm%.3f.txt", JH, Jafm, rank)
        fname_spin  = @sprintf("MC_spin_configurations_JH%.3f_Jafm%.3f.jld2", JH, Jafm, rank)

        open(fname_steps, "w") do io
            println(io, "# T step E accept")
            for T in reverse(Temps)
                Elist = results_energy[T]
                Alist = results_acc[T]
                for i in eachindex(Elist)
                    @printf(io, "%.6f %d %.8f %.6f\n", T, i, Elist[i], Alist[i])
                end
            end
        end

        open(fname_avg, "w") do io
            println(io, "# T <E> <M> <M_AF> <accept>")
            for T in reverse(Temps)
                E = results_energy[T]
                M = results_mag[T]
                AF = results_mAF[T]
                A = results_acc[T]
                @printf(io, "%.6f %.8f %.8f %.8f %.8f\n",
                        T, mean(E), mean(M), mean(AF), mean(A))
            end
        end

        @save fname_spin spin_configs Temps
    end

    MPI.Barrier(comm)
    if rank == 0
        println("[Main] All simulations finished at ", Dates.now())
    end
    MPI.Finalize()
end
main()
