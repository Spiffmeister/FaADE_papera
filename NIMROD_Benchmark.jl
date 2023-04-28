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
nx = ny = 32

Dom = Grid2D(𝒟x,𝒟y,nx,ny)

order = 2

# Homogeneous boundary conditions
BoundaryLeft    = Boundary(Dirichlet,(y,t) -> 0.0, Left, 1)
BoundaryRight   = Boundary(Dirichlet,(y,t) -> 0.0, Right, 1)
BoundaryUp      = Boundary(Dirichlet,(x,t) -> 0.0, Up, 2)
BoundaryDown    = Boundary(Dirichlet,(x,t) -> 0.0, Down, 2)


# Initial condition
u₀(x,y) = cos(π*x)*cos(π*y)

# Perpendicular diffusion coefficient
k(x,y) = 1.0e-2

# Build PDE problem
P = VariableCoefficientPDE2D(u₀,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)

# Time domain
Δt = 0.01Dom.Δx^2
t_f = 10.0




#===
    CONSTRUCT GRID
===#
function B(X,x,p,t)
    X[1] = -π*cos(π*x[1])*sin(π*x[2])
    X[2] = π*sin(π*x[1])*cos(π*x[2])
    # X[3] = 0.0
end

# B(x) = Point2f([-π*cos(π*x[1])*sin(π*x[2])
# π*sin(π*x[1])*cos(π*x[2])
# ])

# P = ODEProblem(B,[0.2,0.0],(0.0,2π))
# solve(P,Rodas4P())

gdata   = construct_grid(B,Dom,[-2π,2π],ymode=:stop)
Pfn     = generate_parallel_penalty(gdata,Dom,order)


planeerror = zeros(nx,ny);
for i = 1:nx
    for j = 1:ny
        planeerror[i,j] = norm([gdata.Fplane[i,j][1] - gdata.plane[i,j][1], gdata.Fplane[i,j][1] - gdata.plane[i,j][1]])
    end
end





# Solve
@time soln = solve(P,Dom,Δt,2.1Δt,:cgie,adaptive=true,penalty_func=Pfn,source=F)
@time soln = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn,source=F)


# Exact solution
T(x,y,t) = (1.0 - exp(-2.0*k(x,y)*π^2*t) )/( k(x,y) )*Ψ(x,y)

T_exact = zeros(Dom.nx,Dom.ny)
for j = 1:ny
    for i = 1:nx
        T_exact[i,j] = T(Dom.gridx[i],Dom.gridy[j],t_f)
    end
end




norm(T_exact - soln.u[2])/norm(T_exact)

using Plots
surface(T_exact)
surface(soln.u[2])


