using LinearAlgebra
using DelimitedFiles

push!(LOAD_PATH,"../FaADE")
using FaADE





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
end


N = [17,25,33,41]

k_perp = 1.0
k_para = 1.0

k(x,y) = k_perp
# ttol = 1e-6

T(x,y,t) = (1.0 - exp(-2.0*k_perp*π^2*t) )/( k_perp )*Ψ(x,y)
# T(x,y,t) = 2π^2*t * Ψ(x,y)

# Diffusion coefficient
order = 4
n = 33

nx = ny = n
Dom = Grid2D(𝒟x,𝒟y,nx,ny)

# Build PDE problem
P = VariableCoefficientPDE2D(u₀,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)
# P = VariableCoefficientPDE2D(u₀,k,k,order,PeriodicBoundary(1),PeriodicBoundary(2))

Δt = 0.1Dom.Δx^2 / 3.0
t_f = 1/(2π^2)
# t_f = Δt
nf = round(t_f/Δt)
Δt = t_f/nf

gdata   = construct_grid(B,Dom,[-2.0π,2.0π],ymode=:stop)

# for i = 1:n #remap all points back to themselves
#     gdata.Fplane.x[:,i] = Dom.gridx
#     gdata.Fplane.y[i,:] = Dom.gridy
#     gdata.Bplane.x[:,i] = Dom.gridx
#     gdata.Bplane.y[i,:] = Dom.gridy
# end


Pfn = generate_parallel_penalty(gdata,Dom,order,κ=k_para)#,perp=k_perp)

println("===",k_perp,"===",k_para,"===",order,"===")
println(nx," ",t_f," ",Δt,"     ",nf)

# soln = solve(P,Dom,Δt,2.1Δt,:cgie,adaptive=false,   source=F,penalty_func=Pfn)
soln = solve(P,Dom,Δt,t_f,:cgie,adaptive=false,nf=nf,   source=F)#,penalty_func=Pfn)
# soln = solve(P,Dom,Δt,t_f,:cgie,adaptive=false,Pgrid=gdata,source=F)

T_exact = zeros(Dom.nx,Dom.ny);
for j = 1:ny
    for i = 1:nx
        T_exact[i,j] = T(Dom.gridx[i],Dom.gridy[j],t_f)
    end
end



# println("pollution=",abs(1/k(0.0,0.0) - soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1])*k(0.0,0.0))
# println("yes",(abs(T(0.0,0.0,t_f) - soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1]))/T(0.0,0.0,t_f))
println("rel err=",norm(T_exact .- soln.u[2])/norm(T_exact))



# XY = (16,17)
# sqrt(gdata.Fplane.x[XY[1],XY[2]]^2 + gdata.Fplane.x[XY[1],XY[2]]^2)



using Plots
# surface(T_exact)
# surface(soln.u[2])

# plot(soln.u[2][floor(Int,nx/2)+1,:]); plot!(T_exact[floor(Int,nx/2)+1,:]); scatter!(soln.u[2][floor(Int,nx/2)+1,:])


plot!(abs.(T_exact[floor(Int,nx/2)+1,:] - soln.u[2][floor(Int,nx/2)+1,:]))

# gdata.Fplane.ox[17,5],gdata.Fplane.oy[17,5]
# gdata.Fplane.x[17,5] ,gdata.Fplane.y[17,5]
