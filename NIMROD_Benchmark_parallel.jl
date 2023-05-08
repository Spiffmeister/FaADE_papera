using LinearAlgebra
using DelimitedFiles

push!(LOAD_PATH,"../SBP_operators")
using SBP_operators





Ψ(x,y) = cos(π*x)*cos(π*y)

# Diffusion coefficient
κ_para = 1.0e3
k(x,y) = 1.0 #perpendicular diffusion
ttol = 1e-6


# Domain
𝒟x = [-0.5,0.5]
𝒟y = [-0.5,0.5]

# Homogeneous boundary conditions
BoundaryLeft    = Boundary(Dirichlet,(y,t) -> cos(0.5π)*cos(π*y)    , Left, 1)
BoundaryRight   = Boundary(Dirichlet,(y,t) -> cos(-0.5π)*cos(π*y)   , Right, 1)
BoundaryUp      = Boundary(Dirichlet,(x,t) -> cos(π*x)*cos(0.5π)    , Up, 2)
BoundaryDown    = Boundary(Dirichlet,(x,t) -> cos(π*x)*cos(-0.5π)   , Down, 2)

# Initial condition
u₀(x,y) = 0.0
# Source term
F(x,y,t) = 2π^2*cos(π*x)*cos(π*y)
# Magnetic field
function B(X,x,p,t)
    X[1] = -π*cos(π*x[1])*sin(π*x[2])
    X[2] = π*sin(π*x[1])*cos(π*x[2])
    # X[3] = 0.0
end
# Exact solution
T(x,y,t) = (1.0 - exp(-2.0*k(x,y)*π^2*t) )/( k(x,y) )*Ψ(x,y)


N = [17,25,33,41]
# n = 17


for k_para in [1.0,1.0e3,1.0e6,1.0e9]
    κ_para = k_para
    k(x,y) = 1.0 #perpendicular diffusion
    ttol = 1e-6
    for order in [2,4]
        pollution = []
        rel_error = []
        for n in N

            
            nx = ny = n
            Dom = Grid2D(𝒟x,𝒟y,nx,ny)
            
            # Build PDE problem
            P = VariableCoefficientPDE2D(u₀,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)

            # Time domain
            Δt = 0.1Dom.Δx^2
            # t_f = 1/(k(0.0,0.0) * 2 * π^2) * log(1/ttol)
            t_f = 1.0

            gdata   = construct_grid(B,Dom,[-2.0,2.0],ymode=:stop)
            Pfn = generate_parallel_penalty(gdata,Dom,order,κ=κ_para) # Generate a parallel penalty with a modified penalty parameter


            println(nx," ",t_f)

            soln = solve(P,Dom,Δt,2.1Δt,:cgie,adaptive=false,Pgrid=gdata,source=F)
            soln = solve(P,Dom,Δt,t_f,:cgie,adaptive=false,Pgrid=gdata,source=F)

            T_exact = zeros(Dom.nx,Dom.ny);
            for j = 1:ny
                for i = 1:nx
                    T_exact[i,j] = T(Dom.gridx[i],Dom.gridy[j],1.0)
                end
            end
            

            push!(pollution, abs(1/k(0.0,0.0) - soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1]))

            push!(rel_error, norm(T_exact .- soln.u[2])/norm(T_exact))

        end

        open(string("NB_kpara_",κ_para,"_pollution_O",order,".csv"),"w") do io
            writedlm(io,[N pollution])
        end

        open(string("NB_kpara_",κ_para,"_relerr_O",order,".csv"),"w") do io
            writedlm(io,[N rel_error])
        end
    end
end
# using Plots
# surface(T_exact)
# surface(soln.u[2])

# Solve




# println(T(0.0,0.0,t_f) .- soln.u[2][argmin(abs.(Dom.gridx)),argmin(abs.(Dom.gridy))])

# println(soln.u[2][argmin(abs.(Dom.gridx)),argmin(abs.(Dom.gridy))] .- k(0.0,0.0))


# using Plots
# surface(T_exact)
# surface(soln.u[2])


# ϵ = zeros(nruns)
# for i = 1:nruns
#     ϵ[i] = norm( 1/k(0,0) - solns[i].soln.u[2][ argmin(abs.(solns[i].Dom.gridx)), argmin(abs.(solns[i].Dom.gridy)) ] )
# end

