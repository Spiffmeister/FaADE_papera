"""
Section 4.2 - Effective penalty parameter τ_eff in NIMROD benchmark
"""
using LinearAlgebra
using DelimitedFiles
using CSV
using DataFrames
using Statistics

using FaADE



θ = 0.5

Ψ(x,y) = cos(π*x)*cos(π*y)


# Domain
𝒟x = [-0.5,0.5]
𝒟y = [-0.5,0.5]


# Initial condition
u₀(x,y) = 0.0
# Source term
F(X,t) = 2π^2*cos(π*X[1])*cos(π*X[2])
# Magnetic field for FLT
function B(X,x,p,t)
    X[1] = -π*cos(π*x[1])*sin(π*x[2])
    X[2] = π*sin(π*x[1])*cos(π*x[2])
end



N = [25,51]

KPEXP = [0,3,6,9]
KP = 10 .^KPEXP


coord = :Cartesian

for dt in [0.1,0.05,0.02]
    # k_para = 1.0
    k = k_perp = 1.0
    T(x,y,t) = (1.0 - exp(-2.0*k_perp*π^2*t) )/( k_perp )*Ψ(x,y)
    # Diffusion coefficient


    Δt = dt
    t_f = 1.0
    nf = round(t_f/Δt)
    Δt = t_f/nf

    dictout = Dict{String,Any}()
    T = LinRange(0.0,1.0,Int(nf))
    dictout["t"] = T
    
    for order in [2]

        for n in N
        
            println(" --- order=",order," --- θ=",θ," ---")
            for EXP in KPEXP

                k_para = 10.0^EXP
                nx = ny = n
                Dom = Grid2D(𝒟x,𝒟y,nx,ny)

                # Homogeneous boundary conditions
                BoundaryLeft    = FaADE.SATs.SAT_Dirichlet((y,t) -> cos(0.5π)*cos(π*y)    , Dom.Δx, Left,   order, Dom.Δy, coord)
                BoundaryRight   = FaADE.SATs.SAT_Dirichlet((y,t) -> cos(-0.5π)*cos(π*y)   , Dom.Δx, Right,  order, Dom.Δy, coord)
                BoundaryUp      = FaADE.SATs.SAT_Dirichlet((x,t) -> cos(π*x)*cos(0.5π)    , Dom.Δy, Up,     order, Dom.Δx, coord)
                BoundaryDown    = FaADE.SATs.SAT_Dirichlet((x,t) -> cos(π*x)*cos(-0.5π)   , Dom.Δy, Down,   order, Dom.Δx, coord)

                BC = FaADE.Inputs.SATBoundaries(BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)

                
                # Time setup
                # Δt = 0.1Dom.Δx

                
                gdata   = construct_grid(B,Dom,[-2.0π,2.0π],ymode=:stop)
                PData   = ParallelData(gdata,Dom,order,κ=k_para) # Generate a parallel penalty with a modified penalty parameter
                
                # Build PDE problem
                P = newProblem2D(order,u₀,k,k,Dom,BC,F,PData)
                
                soln = solve(P,Dom,Δt,1.1Δt,solver=:theta, θ=θ)
                soln = solve(P,Dom,Δt,t_f,  solver=:theta, θ=θ)
                
                T = LinRange(0.0,t_f,Int(nf))


                
                dictout["tau K$EXP O$order N$n"] = soln.τ_hist
            end
        end
    end
    df = DataFrame(dictout)
    CSV.write(string("NIMROD_out/exp2/NB_tau_dt",dt,".csv"),df)
end

