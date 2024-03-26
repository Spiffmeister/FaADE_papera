"""
Section 4.2 - Convergence rates for NIMROD benchmark
"""
using LinearAlgebra
using Statistics
using DelimitedFiles
using CSV
using BasicInterpolators
using DataFrames

# push!(LOAD_PATH,"../FaADE")
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
# Magnetic field
# function B(X,x,p,t)
#     X[1] = π*cos(π*x[1])*sin(π*x[2])
#     X[2] = -π*sin(π*x[1])*cos(π*x[2])
# end

function B(X,x,p,t)
    bn = π * sqrt(abs(cos(x[1]*π)*sin(x[2]*π))^2 + abs(sin(x[1]*π)*cos(x[2]*π))^2)
    X[1] = π*cos(π*x[1])*sin(π*x[2])/bn
    X[2] = -π*sin(π*x[1])*cos(π*x[2])/bn
    if (x[1] == 0.5) && (x[2] == 0.5)
        X[1] = 0.0
        X[2] = -1.0
    elseif (x[1] == 0.5) && (x[2] == -0.5)
        X[1] = -1.0
        X[2] = 0.0
    elseif (x[1] == -0.5) && (x[2] == -0.5)
        X[1] = 0.0
        X[2] = 1.0
    elseif (x[1] == -0.5) && (x[2] == 0.5)
        X[1] = 1.0
        X[2] = 0.0
    elseif (x[1] == 0.0) && (x[2] == 0.0)
        X[1] = 0.0
        X[2] = 0.0
    end
    # X[3] = 0.0
end

MagField(X,t) = [
    π*cos(π*X[1])*sin(π*X[2]),
    -π*sin(π*X[1])*cos(π*X[2]),
    0.0
]
# Exact solution
T(x,y,t) = (1.0 - exp(-2.0*π^2*t) )*Ψ(x,y) # k_perp = 1


N = collect(21:10:101)
# N = collect(22:10:102)

coord = :Cartesian

for order in [2,4]
# for order in [4]

    dictout = Dict{String,Any}()
    dictout["N"] = N

    for EXP in [0.0,3.0,5.0,6.0,7.0,9.0,10.0]
    # for k_para in [0.0,1e3,1e5,1e6,1e7,1e9,1e10]
    # for k_para in [1e12]
        k = 1.0 #perpendicular diffusion
        k_para = 10^EXP

        pollution = []
        pollution_time = []
        rel_error = []
        abs_error = []
        tau_hist = []
        println("===PARA=",k_para,"===ORDER=",order,"===")
        for n in N

            nx = ny = n
            Dom = Grid2D(𝒟x,𝒟y,nx,ny)

            # Homogeneous boundary conditions
            BoundaryLeft    = FaADE.SATs.SAT_Dirichlet((y,t) -> cos(0.5π)*cos(π*y)    , Dom.Δx, Left,   order, Dom.Δy, coord)
            BoundaryRight   = FaADE.SATs.SAT_Dirichlet((y,t) -> cos(-0.5π)*cos(π*y)   , Dom.Δx, Right,  order, Dom.Δy, coord)
            BoundaryUp      = FaADE.SATs.SAT_Dirichlet((x,t) -> cos(π*x)*cos(0.5π)    , Dom.Δy, Up,     order, Dom.Δx, coord)
            BoundaryDown    = FaADE.SATs.SAT_Dirichlet((x,t) -> cos(π*x)*cos(-0.5π)   , Dom.Δy, Down,   order, Dom.Δx, coord)

            BC = FaADE.Inputs.SATBoundaries(BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)
 

            gdata   = construct_grid(B,Dom,[-1.0,1.0],ymode=:stop)
            PData   = ParallelData(gdata,Dom,order,κ=k_para)#,B=MagField) # Generate a parallel penalty with a modified penalty parameter
            
            # Build PDE problem
            P = newProblem2D(order,u₀,k,k,Dom,BC,F,PData)

            # Time domain
            Δt = 0.1Dom.Δx^2
            t_f = 0.1
            nf = round(t_f/Δt)
            Δt = t_f/nf

            soln = solve(P,Dom,Δt,2.1Δt,solver=:theta, θ=θ)
            soln = solve(P,Dom,Δt,t_f,  solver=:theta, θ=θ)
            # println(nx,"    t_f=",t_f,"    t_f-t=",t_f-soln.t[2],"     Δt=",Δt,"   nf=",nf)

            
            T_exact = zeros(Dom.nx,Dom.ny);
            for I in eachindex(Dom)
                T_exact[I] = T(Dom[I]...,t_f)
            end


            # Hx  = FaADE.Derivatives.DiagonalH(order,Dom.Δx,nx)
            # Hy  = FaADE.Derivatives.DiagonalH(order,Dom.Δy,ny)
            # H = FaADE.Derivatives.CompositeH(Hx,Hy)
            # tmp = T_exact .- soln.u[2]
            # abs_err = FaADE.Derivatives.mul!(tmp,H,tmp)
            # denom = FaADE.Derivatives.mul!(T_exact,H,T_exact)
            
            tmp = T_exact .- soln.u[2]
            # IH = FaADE.Derivatives.innerH(Dom.Δx,Dom.Δy,nx,ny,order)
            # RE = sqrt(IH(tmp,tmp))/sqrt(IH(T_exact,T_exact))

            RE = norm(tmp)/norm(T_exact)



            if iseven(nx)
                tmpI = BicubicInterpolator(Dom.gridx[:,1],Dom.gridy[1,:],soln.u[2])
                push!(pollution,abs(T(0.0,0.0,t_f) - tmpI(0.0,0.0)))
            else
                push!(pollution, abs(1 - soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1]))
            end
            # push!(pollution, abs(1 - soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1]))
            push!(pollution_time,abs(T(0.0,0.0,t_f) - soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1]))
            # push!(rel_error, norm(T_exact .- soln.u[2])/norm(T_exact))
            push!(rel_error, RE)
            push!(abs_error,norm(tmp))
            push!(tau_hist, mean(soln.τ_hist))
            # println("poll=",pollution[end]," relerr=",rel_error[end]," abserr=",norm(T_exact .- soln.u[2])*Dom.Δx*Dom.Δy)

        end

        # dictout[string("poll ",k_para)] = pollution
        dictout[string("rel ",EXP)] = rel_error
        dictout[string("abs ",EXP)] = abs_error
        # dictout[string("tau ",k_para)] = tau_hist

        # open(string("NIMROD_out/withB/NB_k",κ_para,"_O",order,"_theta",θ,".csv"),"w") do io
        #     writedlm(io,[N pollution rel_error abs_error])
        # end

        conv_rate = log.(rel_error[1:end-1]./rel_error[2:end]) ./ log.( (1 ./ (N[1:end-1].-1))./(1 ./ (N[2:end].-1) ))
        println("abs error ",abs_error)
        println("rel error ",rel_error)
        println("conv rates ",conv_rate)
        # println("Mean tau ",tau_hist)


    end

    df = DataFrame(dictout)

    CSV.write("NIMROD_out/exp2_even/NB_O$(order)_theta$(θ).csv",df)

end
