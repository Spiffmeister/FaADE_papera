using LinearAlgebra

cd("..")
push!(LOAD_PATH,"./SBP_operators")
using SBP_operators





Ψ(x,y) = cos(π*x)*cos(π*y)


# Source term
F(x,y,t) = 2π^2*cos(π*x)*cos(π*y)

# Domain
𝒟x = [-0.5,0.5]
𝒟y = [-0.5,0.5]
nx = ny = 33

Dom = Grid2D(𝒟x,𝒟y,nx,ny)

order = 2

# Homogeneous boundary conditions
BoundaryLeft    = Boundary(Dirichlet,(y,t) -> cos(0.5π)*cos(π*y)    , Left, 1)
BoundaryRight   = Boundary(Dirichlet,(y,t) -> cos(-0.5π)*cos(π*y)   , Right, 1)
BoundaryUp      = Boundary(Dirichlet,(x,t) -> cos(π*x)*cos(0.5π)    , Up, 2)
BoundaryDown    = Boundary(Dirichlet,(x,t) -> cos(π*x)*cos(-0.5π)   , Down, 2)


# Initial condition
u₀(x,y) = cos(π*x)*cos(π*y)

# Perpendicular diffusion coefficient
k(x,y) = 1.0

# Build PDE problem
P = VariableCoefficientPDE2D(u₀,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)

# Time domain
Δt = 0.1Dom.Δx^2
t_f = 10.0




#===
    CONSTRUCT GRID
===#
function B(X,x,p,t)
    X[1] = -π*cos(π*x[1])*sin(π*x[2])
    X[2] = π*sin(π*x[1])*cos(π*x[2])
    # X[3] = 0.0
end

gdata   = construct_grid(B,Dom,[-2.0,2.0],ymode=:stop)
Pfn     = generate_parallel_penalty(gdata,Dom,order)


# Solve
@time soln = solve(P,Dom,Δt,2.1Δt,:cgie,adaptive=false,Pgrid=gdata,source=F)
@time soln = solve(P,Dom,Δt,t_f,:cgie,adaptive=false,Pgrid=gdata,source=F)


# Exact solution
T(x,y,t) = (1.0 - exp(-2.0*k(x,y)*π^2*t) )/( k(x,y) )*Ψ(x,y)

T_exact = zeros(Dom.nx,Dom.ny)
for j = 1:ny
    for i = 1:nx
        T_exact[i,j] = T(Dom.gridx[i],Dom.gridy[j],t_f)
    end
end


println(T(0.0,0.0,t_f) .- soln.u[2][argmin(abs.(Dom.gridx)),argmin(abs.(Dom.gridy))])

println(soln.u[2][argmin(abs.(Dom.gridx)),argmin(abs.(Dom.gridy))] .- k(0.0,0.0))


using Plots
surface(T_exact)
surface(soln.u[2])






#=

ϵ = zeros(nruns)
for i = 1:nruns
    ϵ[i] = norm( 1/k(0,0) - solns[i].soln.u[2][ argmin(abs.(solns[i].Dom.gridx)), argmin(abs.(solns[i].Dom.gridy)) ] )
end

using DelimitedFiles
nameappend=string("k=",k(0,0))
open(string("NIMROD_benchmark",nameappend,".csv"),"w") do io
    writedlm(io,[N ϵ])
end

=#