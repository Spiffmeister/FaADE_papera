using LinearAlgebra

push!(LOAD_PATH,"../SBP_operators")
push!(LOAD_PATH,"../plas_diff")
using SBP_operators
using plas_diff








# Flux function
ψ(r,θ;rₛ=0.7,δ=0.005) = (r - rₛ)^2 + δ*r^2*(1-r^4)*cos(θ)

# Source term
S(r,θ,t) = 4*(1-r^2)^8

# Initial condition
u0(x,y) = 0.0



###
Dx = [0.0,1.0]
Dy = [0.0,2π]
nx = 51
ny = 51
Dom = Grid2D(Dx,Dy,nx,ny)


order = 2

Δt = 0.1Dom.Δx^2
t_f = 10.0

println("Parallel grid construction")
#= MAGNETIC FIELD =#
# ϵ = 0.0 #Perturbation parameter
params = (δ=0.005, xₛ=0.7)
function B(X,x::Array{Float64},p,t)
    X[1] = p.δ*x[1]^2 * (1.0-x[1]^4) * sin(x[2])
    X[2] = 2.0*(x[1] - p.xₛ) + p.δ*(2.0*x[1] - 6*x[1]^5)*cos(x[2])
end
dH(X,x,p,t) = χ_h!(X,x,params,t)
@time PGrid = SBP_operators.construct_grid(dH,Dom,[-2π,2π])
Pfn = SBP_operators.generate_parallel_penalty(PGrid,Dom,order)



BoundaryLeft = Boundary(Dirichlet,(y,t) -> 0.0,SBP_operators.Left,1)
BoundaryRight = Boundary(Dirichlet,(y,t) -> 0.0,SBP_operators.Right,1)
BoundaryUpDown = PeriodicBoundary(2)


# println("Poincare construction")


println("Begin solve")
k(x,y) = 1.0e-6
P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)
@time soln1   = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn,source=S)


#=
k(x,y) = 1.0e-1
P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)
soln2   = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn)

k(x,y) = 1.0e-2
P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)
soln3   = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn)

k(x,y) = 0.0
P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)
soln4  = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn)
=#







using GLMakie
# using CairoMakie
using Contour
using Interpolations


xvals = LinRange(0.0,1.0,101);

function findcontours(xvals,soln,Dom)
    itp = LinearInterpolation(Dom.gridx,soln.u[2][:,1])
    uvals = itp(xvals)
    return uvals
end


c1  = findcontours(xvals,soln1,Dom);
# c2  = findcontours(xvals,soln2,Dom);
# c3  = findcontours(xvals,soln3,Dom);
# c4  = findcontours(xvals,soln4,Dom);







P = scatter(pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)
ylims!(0.0,1.0)
contour!(Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=c1[1:40], linewidth=3.0)
contour!(Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=c1[70:end], linewidth=3.0)


contour!(Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=[c1[48]], linewidth=3.0)
contour!(Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=[c1[69]], linewidth=3.0)


contour!(Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=[(c1[68]+c1[69])/2.0], linewidth=3.0)

contour!(Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=[c1[70]-1e-3], linewidth=3.0)
contour!(Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=[c1[69]+5e-3], linewidth=3.0)


contour!(Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=[c1old[69]-1e-2], linewidth=3.0)
contour!(Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=[0.6], linewidth=3.0)

for δ in LinRange(0.58,0.6,21)
    contour!(Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=[δ], linewidth=3.0)
end


#=
con1 =  Contour.contour(Dom.gridx,Dom.gridy,soln1.u[2],0.495)
con6 =  Contour.contour(Dom.gridx,Dom.gridy,soln6.u[2],0.495)
con8 =  Contour.contour(Dom.gridx,Dom.gridy,soln8.u[2],0.495)
con10 = Contour.contour(Dom.gridx,Dom.gridy,soln10.u[2],c10[1])
con0 =  Contour.contour(Dom.gridx,Dom.gridy,soln0.u[2],c0[1])

con1valx = [x[1] for x in con1.lines[1].vertices];
con1valy = [y[2] for y in con1.lines[1].vertices];
con6valx = [x[1] for x in con6.lines[1].vertices];
con6valy = [y[2] for y in con6.lines[1].vertices];
con8valx = [x[1] for x in con8.lines[1].vertices];
con8valy = [y[2] for y in con8.lines[1].vertices];

con1valx - con6valx
con1valy - con6valy
con6valx - con8valx
con6valy - con8valy
=#

#=
F = Figure()
GA = F[1,1] = GridLayout()
# F = Figure(resolution=(1600,1200),fontsize=40)

Ax1 = Axis(GA[1,1],                  ylabel=L"\psi", xlabelsize=50,ylabelsize=50);
Ax2 = Axis(GA[1,2],                                  xlabelsize=50,ylabelsize=50);
Ax3 = Axis(GA[2,1],xlabel=L"\theta", ylabel=L"\psi", xlabelsize=50,ylabelsize=50);
Ax4 = Axis(GA[2,2],xlabel=L"\theta",                 xlabelsize=50,ylabelsize=50);


linkyaxes!(Ax1, Ax2); linkyaxes!(Ax1, Ax3); linkyaxes!(Ax1, Ax4)


# Add Poincare plots
scatter!(Ax1,pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)
scatter!(Ax2,pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)
scatter!(Ax3,pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)
scatter!(Ax4,pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)

con1    = contour!(Ax1,Dom.gridy,Dom.gridx,soln1.u[2]'  ,levels=c1, linewidth=3.0)
con2    = contour!(Ax2,Dom.gridy,Dom.gridx,soln2.u[2]'  ,levels=c2, linewidth=3.0)
con3    = contour!(Ax3,Dom.gridy,Dom.gridx,soln3.u[2]'  ,levels=c3, linewidth=3.0)
con4    = contour!(Ax4,Dom.gridy,Dom.gridx,soln4.u[2]'  ,levels=c4, linewidth=3.0)

colgap!(GA,5)
rowgap!(GA,5)


Colorbar(F[1,2],con6,label=L"u")#,colorrange=(0.0,1.0))



hidexdecorations!(Ax1,ticks=false)
hidexdecorations!(Ax2,ticks=false)

hideydecorations!(Ax2,ticks=false)
hideydecorations!(Ax4,ticks=false)
limits!(Ax1,-π,π,0.0,1.0)
limits!(Ax2,-π,π,0.0,1.0)
limits!(Ax3,-π,π,0.0,1.0)
limits!(Ax4,-π,π,0.0,1.0)






Ax3.xticks = (-π:π/2:π,["π", "-π/2", "0", "π/2", "π"])
Ax4.xticks = (-π:π/2:π,["π", "-π/2", "0", "π/2", "π"])
Ax1.yticks = (0.0:0.25:1.0,["","0.25","0.5","0.75",""])
Ax3.yticks = (0.0:0.25:1.0,["","0.25","0.5","0.75",""])
=#

#=




surface(Dom.gridx*pi,Dom.gridy,soln1.u[2])
wireframe!(Dom.gridx*pi,Dom.gridy,soln1.u[2],color=(:black,0.3),transparancy=true)



P = scatter(pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)
ylims!(0.0,1.0)
ptrace1 = plas_diff.tracer(χ_h!,params,100,[1.0,0.45])
P1 = scatter!(ptrace1[1],ptrace1[2],markersize=10.0)


name = string("./FieldFeatures_k",k(0.0,0.0),".pdf")
save(name, F)#, resolution=(1600,1200), transparency=true)

=#