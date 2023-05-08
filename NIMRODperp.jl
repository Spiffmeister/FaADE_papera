using LinearAlgebra
using DelimitedFiles

push!(LOAD_PATH,"../SBP_operators")
using SBP_operators





Ψ(x,y) = cos(π*x)*cos(π*y)


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


N = [17,25,33,41]
# k_perp = 1e-3

k_perp = 1e-6

println("===",k_perp,"===")
κ = k_perp
k(x,y) = κ
ttol = 1e-6
T(x,y,t) = (1.0 - exp(-2.0*k(x,y)*π^2*t) )/( k_perp )*Ψ(x,y)
# Diffusion coefficient
order = 2
n = 33
    
nx = ny = n
Dom = Grid2D(𝒟x,𝒟y,nx,ny)

# Build PDE problem
P = VariableCoefficientPDE2D(u₀,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)

# Time domain
Δt = 0.01Dom.Δx^2/k(0.0,0.0)
t_f = 1/(k(0.0,0.0) * 2 * π^2) * log(1/ttol)

gdata   = construct_grid(B,Dom,[-2.0,2.0],ymode=:stop)

println(nx," ",t_f)

soln = solve(P,Dom,Δt,2.1Δt,:cgie,adaptive=false,Pgrid=gdata)#,source=F)
soln = solve(P,Dom,Δt,t_f,:cgie,adaptive=false,Pgrid=gdata,source=F)

T_exact = zeros(Dom.nx,Dom.ny)
for j = 1:ny
    for i = 1:nx
        T_exact[i,j] = T(Dom.gridx[i],Dom.gridy[j],t_f)
    end
end



println("pollution=",abs(1/k(0.0,0.0) - soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1]))
abs(T(0.0,0.0,t_f) - soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1])

println("rel err=",norm(T_exact .- soln.u[2])/norm(T_exact))





# using Plots
# surface(T_exact)
# surface(soln.u[2])

