using LinearAlgebra

push!(LOAD_PATH,"../SBP_operators")
push!(LOAD_PATH,"../plas_diff")
using SBP_operators
using plas_diff




###
Dx = [0.0,1.0]
Dy = [-π,π]
nx = 101
ny = 151
Dom = Grid2D(Dx,Dy,nx,ny)


order = 2

Δt = 0.1Dom.Δx^2
t_f = 1.0

k(x,y) = 1.0


#= MAGNETIC FIELD =#
ϵ = 2.1e-3 + 5e-3 #Perturbation parameter
# ϵ = 0.0 #Perturbation parameter
params = (ϵₘₙ = [ϵ/2., ϵ/3.], m=[2.0, 3.0], n=[1.0, 2.0])
function χ_h!(χ,x::Array{Float64},p,t)
    # Hamiltons equations for the field-line Hamiltonian
    # H = ψ²/2 - ∑ₘₙ ϵₘₙ(cos(mθ - nζ))
    χ[2] = x[1] #p_1            qdot        θ
    χ[1] = -sum(p.ϵₘₙ .*(sin.(p.m*x[2] - p.n*t) .* p.m)) #q_1        pdot        ψ
end
dH(X,x,p,t) = χ_h!(X,x,params,t)
PGrid = SBP_operators.construct_grid(dH,Dom,[-2π,2π])
Pfn = SBP_operators.generate_parallel_penalty(PGrid,Dom,order)


u0(x,y) = x

BoundaryLeft = Boundary(Dirichlet,(y,t) -> 0.0,SBP_operators.Left,1)
BoundaryRight = Boundary(Dirichlet,(y,t) -> 1.0,SBP_operators.Right,1)
BoundaryUpDown = PeriodicBoundary(2)



params = plas_diff.SampleFields.H_params([ϵ/2., ϵ/3.], [2.0, 3.0], [1.0, 2.0])
function χ_h!(χ,x::Array{Float64},p,t)
    # Hamiltons equations for the field-line Hamiltonian
    # H = ψ²/2 - ∑ₘₙ ϵₘₙ(cos(mθ - nζ))
    χ[1] = x[2] #p_1            qdot        θ
    χ[2] = -sum(p.ϵₘₙ .*(sin.(p.m*x[1] - p.n*t) .* p.m)) #q_1        pdot        ψ
end
pdata = plas_diff.poincare(χ_h!,params,N_trajs=500,N_orbs=200,x=Dx,y=Dy)



P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)
soln1   = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn)

k(x,y) = 1.0e-1
P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)
soln2   = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn)

k(x,y) = 1.0e-5
P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)
soln3   = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn)

k(x,y) = 0.0
P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)
soln4  = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn)








using GLMakie
# using CairoMakie
using Contour
using Interpolations


xvals = LinRange(0.0,1.0,21);

function findcontours(xvals,soln,Dom)
    itp = LinearInterpolation(Dom.gridx,soln.u[2][:,1])
    uvals = itp(xvals)
    return uvals
end


c1  = findcontours(xvals,soln1,Dom);
c2  = findcontours(xvals,soln2,Dom);
c3  = findcontours(xvals,soln3,Dom);
c4  = findcontours(xvals,soln4,Dom);







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


#=

name = string("./FieldFeatures_k",k(0.0,0.0),".pdf")
save(name, F)#, resolution=(1600,1200), transparency=true)

=#