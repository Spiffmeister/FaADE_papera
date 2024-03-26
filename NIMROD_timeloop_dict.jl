"""
Section 4.2 - Temporal convergence rates for NIMROD benchmark
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
MagField(X,t) = [
    -π*cos(π*X[1])*sin(π*X[2]),
    π*sin(π*X[1])*cos(π*X[2]),
    0.0
]
# Exact solution


# M = [0.1,0.075,0.05,0.025,0.01]
M = [0.1/2^i for i in 0:6]


n = 201 #fixed grid resolution

coord = :Cartesian

for order in [2,4]
    # k_para = 1.0
    k = k_perp = 1.0
    T(x,y,t) = (1.0 - exp(-2.0*k_perp*π^2*t) )/( k_perp )*Ψ(x,y)
    # Diffusion coefficient

    dictout = Dict{String,Any}()
    dictout["dt"] = M

    for k_para in [0.0,1e3,1e5,1e6,1e7,1e9,1e12]
        pollution = []
        rel_error = []
        abs_error = []

        comp_error = []
        comp_poll = []

        tau_hist = []
        
        println("--- k_para=",k_para," --- order=",order," --- θ=",θ," ---")
        # for dt in [0.1,0.075,0.05,0.025,0.01]
        for dt in M
            nx = ny = n
            Dom = Grid2D(𝒟x,𝒟y,nx,ny)

            # Homogeneous boundary conditions
            BoundaryLeft    = FaADE.SATs.SAT_Dirichlet((y,t) -> cos(0.5π)*cos(π*y)    , Dom.Δx, Left,   order, Dom.Δy, coord)
            BoundaryRight   = FaADE.SATs.SAT_Dirichlet((y,t) -> cos(-0.5π)*cos(π*y)   , Dom.Δx, Right,  order, Dom.Δy, coord)
            BoundaryUp      = FaADE.SATs.SAT_Dirichlet((x,t) -> cos(π*x)*cos(0.5π)    , Dom.Δy, Up,     order, Dom.Δx, coord)
            BoundaryDown    = FaADE.SATs.SAT_Dirichlet((x,t) -> cos(π*x)*cos(-0.5π)   , Dom.Δy, Down,   order, Dom.Δx, coord)

            BC = FaADE.Inputs.SATBoundaries(BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)

            
            # Time setup
            Δt = dt
            t_f = 0.1
            nf = round(t_f/Δt)
            Δt = t_f/nf
            
            gdata   = construct_grid(B,Dom,[-2.0π,2.0π],ymode=:stop)
            PData   = ParallelData(gdata,Dom,order,κ=k_para)#,B=MagField) # Generate a parallel penalty with a modified penalty parameter
            
            # Build PDE problem
            P = newProblem2D(order,u₀,k,k,Dom,BC,F,PData)
            
            soln = solve(P,Dom,Δt,1.1Δt,solver=:theta, θ=θ)
            soln = solve(P,Dom,Δt,t_f,  solver=:theta, θ=θ)
            
            # Solution without parallel operator
            Pwo = newProblem2D(order,u₀,k,k,Dom,BC,F,nothing)
            solnwo = solve(P,Dom,Δt,t_f,solver=:theta, θ=θ)

            T_exact = zeros(eltype(Dom),size(Dom));
            for I in eachindex(Dom)
                T_exact[I] = T(Dom[I]...,soln.t[end])
            end

            push!(pollution, abs(1/soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1] - 1))
            push!(rel_error, norm(T_exact .- soln.u[2])/norm(T_exact))
            push!(abs_error, norm(T_exact .- soln.u[2]))
            # compare with and without the parallel operator
            push!(comp_error, norm(soln.u[2] .- solnwo.u[2]))
            push!(comp_poll, abs(pollution[end] - abs(1/solnwo.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1] - 1)))
            push!(tau_hist, mean(soln.τ_hist))

        end

        # dictout[string("poll ",k_para)] = pollution
        dictout[string("rel ",k_para)] = rel_error
        # dictout[string("abs ",k_para)] = abs_error
        # dictout[string("tau ",k_para)] = tau_hist

        # open(string("timeloop/NB_para",k_para,"_n",n,"_O",order,"_theta",θ,".csv"),"w") do io
        #     writedlm(io,[M rel_error abs_error pollution comp_error comp_poll])
        # end

        println("rel error=",rel_error)
        println("conv rates ",log2.(rel_error[1:end-1]./rel_error[2:end]))
        println("pollution error=",pollution)
        println("pollution Δ=",comp_poll)
    end
    df = DataFrame(dictout)
    CSV.write(string("NIMROD_out/exp3/NB_TL_n",n,"_O",order,".csv"),df)
end

