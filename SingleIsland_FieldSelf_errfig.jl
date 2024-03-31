"""
Section 4.3 - Error surface plot and line for single island field
"""
using LinearAlgebra
using Revise
using BasicInterpolators
using FaADE


using JLD2
using DataFrames
using CSV


plot = true
poincare = true
reference = false
save_rates = false

θ = 0.5
order = 2

k_para = 1.0e6
k_perp = 1.0

u₀(x,y) = x
S = nothing

# Domain
Dx(x,nx) = sinh(0.15*x * (nx/51)^1.3)/2sinh(0.15*(nx/51)^1.3) + 0.5
# function Dx(x,nx; B=0.5)
# 	alpha = (50/nx)^2
# 	return B + alpha * sinh(asinh((1-B)/alpha)*x + asinh((0-B)/alpha)*(1-x))
# end
# Dx(x,nx) = x
Dy(y) = y

coord = :Cartesian




δ = 0.05
xₛ = 0.5

function B(X,x,p,t)
    # bn = 1 + abs( δ*x[1]*(x[1]-1)*sin(x[2]) )^2 + abs( 2*x[1] - 2*xₛ + δ*(1-x[1])*cos(x[2]) - δ*x[1]*cos(x[2]) )^2
    # bn = sqrt(bn)
    X[1] = δ*x[1]*(1-x[1])*sin(x[2])#/bn
    X[2] = 2x[1] - 2*xₛ + δ*(1-x[1])*cos(x[2]) - δ*x[1]*cos(x[2])#/bn
    # X[3] = 0.0
end
MagField(x,t) = [
    δ*x[1]*(1-x[1])*sin(x[2]),
    2x[1] - 2*xₛ + δ*(1-x[1])*cos(x[2]) - δ*x[1]*cos(x[2]),
    0.0
]





function sol(nx,ny,order,reference)
    𝒟x,𝒟y= FaADE.Grid.meshgrid(Dx.(LinRange(-1.0,1.0,nx),nx),Dy.(LinRange(0,2π,ny)))
    Dom = Grid2D(𝒟x,𝒟y,ymap=false)

    if reference# || nx==201
       order = 4
    end

    # Homogeneous boundary conditions
    BoundaryLeft    = FaADE.SATs.SAT_Dirichlet((y,t) -> 0.0    , Dom.Δx , Left,  order, Dom.Δy, coord) #x=0
    BoundaryRight   = FaADE.SATs.SAT_Dirichlet((y,t) -> 1.0    , Dom.Δx , Right, order, Dom.Δy, coord) #x=1
    # BoundaryRight   = FaADE.SATs.SAT_Neumann((y,t) -> 0.0    , Dom.Δx , Right,1, order) #x=1
    BoundaryUp      = FaADE.SATs.SAT_Periodic(Dom.Δy,2,order,Up,    Dom.Δx,coord)
    BoundaryDown    = FaADE.SATs.SAT_Periodic(Dom.Δy,2,order,Down,  Dom.Δx,coord)

    BC = FaADE.Inputs.SATBoundaries(BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)

    gdata   = construct_grid(B,Dom,[-2.0π,2.0π],ymode=:period)
    PData   = ParallelData(gdata,Dom,order,κ=k_para)#,B=MagField)

    # Build PDE problem
    P1       = Problem2D(order,u₀,k_perp,k_perp,Dom,BC,S,PData)

    @show Δt = 1.0e-4
    t_f = 1.0e-2
    nf = round(t_f/Δt)
    Δt = t_f/nf

    solve(P1,Dom,Δt,1.1Δt,       solver=:theta,  θ=θ)
    soln = solve(P1,Dom,Δt,t_f,  solver=:theta,  θ=θ)

    return soln, Dom
end



function reconstruct_soln(Dom,refinterp)
    # Itmp = BicubicInterpolator(soln.grid.gridx[:,1],soln.grid.gridy[1,:],soln.u[2])
    # tmpgrid = Grid2D([0.0,1.0],[0.0,2π],soln.grid.nx,soln.grid.ny)
    solncart = zeros(size(Dom));
    for I in eachindex(Dom)
        solncart[I] = refinterp(Dom[I]...)
    end
    return solncart
end



println("start")

soln3, Dom3 = sol(201,201,order,false);


nx = ny = 801
𝒟x,𝒟y= FaADE.Grid.meshgrid(Dx.(LinRange(-1.0,1.0,nx),nx),Dy.(LinRange(0,2π,ny)))
Dom0 = Grid2D(𝒟x,𝒟y,ymap=false)
# @load "SingleIsland_out/SingleIsland K6 O2.jld2" refu
# soln0u = refu
@load "SingleIslandFieldSelf/SingleIslandSelf_stretch k$k_para delta$δ dt1e-4.jld2" soln0u

refinterp = BicubicInterpolator(Dom0.gridx[:,1],Dom0.gridy[1,:],soln0u)

soln3ref = reconstruct_soln(Dom3,refinterp);






using GLMakie; GLMakie.activate!()


f3 = Figure(size=(1200,500),fontsize=20);
f3gl = f3[1,1] = GridLayout()

errline = soln3.u[2][:,101].-soln3ref[:,101];
crng = (minimum(soln3.u[2].-soln3ref),maximum(soln3.u[2].-soln3ref))


axf3a = Axis(f3gl[1,1],ylabel=L"u_{201}-u_{ref}")
axf3a.ylabelsize = 35

include("../FaADE_papera/FieldLines.jl")
poindata = FieldLines.construct_poincare(B,[0.0,1.0],[0,2π],N_trajs=200)


lines!(axf3a,Dom3.gridx[:,1],errline,linewidth=4,color=errline,colorrange=crng)
scatter!(axf3a,poindata.ψ[:],poindata.θ[:]/π * crng[2] .- crng[2],markersize=1.0,color=:black,alpha=1.0)
xlims!(axf3a,0.0,1.0)




axf3b = Axis3(f3gl[1,2],
    zlabelvisible=false,
    zticklabelsvisible=false)

f3scb = surface!(axf3b,Dom3.gridx,Dom3.gridy,(soln3.u[2].-soln3ref),colormap=:viridis,colorrange=crng)
f3scbl = lines!(Dom3.gridx[:,1],Dom3.gridy[:,101],errline,color=:black,linewidth=2,label="Error")

cb3 = Colorbar(f3gl[1,3],f3scb)

axislegend(axf3b,[f3scbl],["Error line"])


colgap!(f3gl,10)




# display(f3,px_per_unit=0.5)


save("SingleIslandFieldSelf/SingleIslandField_error.png",f3,px_per_unit=5.0)



# colgap!(f3gl,10)


#=
using GLMakie
# using CairoMakie

f3 = Figure(size=(1200,800));
f3gl = f3[1,1] = GridLayout()
axf3a = Axis3(f3gl[1,1],protrusions=(0,0,0,20))
axf3b = Axis3(f3gl[1,2],protrusions=(0,0,0,20))
axf3c = Axis3(f3gl[2,1],protrusions=(0,0,0,20))
axf3d = Axis3(f3gl[2,2],protrusions=(0,0,0,20))

crng = (minimum(soln1.u[2].-soln1ref),maximum(soln1.u[2].-soln1ref))
cm = :redsblues

f3sca = surface!(axf3a,Dom1.gridx,Dom1.gridy,(soln1.u[2].-soln1ref),colormap=cm,colorrange=crng)
f3scb = surface!(axf3b,Dom2.gridx,Dom2.gridy,(soln2.u[2].-soln2ref),colormap=cm,colorrange=crng)
f3scc = surface!(axf3c,Dom3.gridx,Dom3.gridy,(soln3.u[2].-soln3ref),colormap=cm,colorrange=crng)
f3scd = surface!(axf3d,Dom4.gridx,Dom4.gridy,(soln4.u[2].-soln4ref),colormap=cm,colorrange=crng)

Colorbar(f3gl[1:2,3],f3sca)

text!(axf3a,0.1,0.9,text=L"n_x=51",space=:relative,fontsize=20)
text!(axf3b,0.1,0.9,text=L"n_x=101",space=:relative,fontsize=20)
text!(axf3c,0.1,0.9,text=L"n_x=201",space=:relative,fontsize=20)
text!(axf3d,0.1,0.9,text=L"n_x=401",space=:relative,fontsize=20)

colgap!(f3gl,0)
rowgap!(f3gl,50)
=#