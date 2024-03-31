"""
Section 4.3 - Single island error
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
save_rates = true

θ = 0.5
order = 4

k_para = 1.0e7
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



# scatter(Dx.(LinRange(-1,1,801),801),LinRange(-1,1,801))
# scatter!(Dx.(LinRange(-1,1,51),51),LinRange(-1,1,51),marker=:x,markersize=10)

# iamthecountilovetocounthaha = zeros(5)
# iamthecountilovetocounthaha[1] = count(i->(0.4≤i≤0.6), Dx.(LinRange(-1,1,51),51))
# iamthecountilovetocounthaha[2] = count(i->(0.4≤i≤0.6), Dx.(LinRange(-1,1,101),101))
# iamthecountilovetocounthaha[3] = count(i->(0.4≤i≤0.6), Dx.(LinRange(-1,1,201),201))
# iamthecountilovetocounthaha[4] = count(i->(0.4≤i≤0.6), Dx.(LinRange(-1,1,401),401))
# iamthecountilovetocounthaha[5] = count(i->(0.4≤i≤0.6), Dx.(LinRange(-1,1,801),801))
# iamthecountilovetocounthaha[2:end]./iamthecountilovetocounthaha[1:end-1]

# Dx.(LinRange(-1,1,801),801)

# sum(isnan.(Dx.(LinRange(-1,1,801),801)))

# DDx(x) = 2*cosh(2x)/2sinh(2)

# 𝒟x = [0.0,1.0]
# 𝒟y = [0.0,2π]
# coord = :Cartesian
# 𝒟x,𝒟y= FaADE.Grid.meshgrid(Dx.(LinRange(-1.0,1.0,nx)),Dy.(LinRange(0,2π,ny)))
# Dom = Grid2D(𝒟x,𝒟y,ymap=false)



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

soln1, Dom1 = sol(51,51,order,false);
soln2, Dom2 = sol(101,101,order,false);
soln3, Dom3 = sol(201,201,order,false);
soln4, Dom4 = sol(301,301,order,false);
soln5, Dom5 = sol(401,401,order,false);
soln6, Dom6 = sol(501,501,order,false);
soln7, Dom7 = sol(601,601,order,false);
soln8, Dom8 = sol(701,701,order,false);

if reference
    println("reference")
    nx = ny = 801
    soln0, Dom0 = sol(nx,ny,order,reference)
    soln0u = soln0.u[2]

    jldsave("SingleIslandFieldSelf/SingleIslandSelf_stretch k$k_para delta$δ dt1e-4.jld2"; soln0u=soln0.u[2], grid0x=Dom0.gridx, grid0y=Dom0.gridy)

else
    nx = ny = 801
    𝒟x,𝒟y= FaADE.Grid.meshgrid(Dx.(LinRange(-1.0,1.0,nx),nx),Dy.(LinRange(0,2π,ny)))
    Dom0 = Grid2D(𝒟x,𝒟y,ymap=false)
    # @load "SingleIsland_out/SingleIsland K6 O2.jld2" refu
    # soln0u = refu
    @load "SingleIslandFieldSelf/SingleIslandSelf_stretch k$k_para delta$δ dt1e-4.jld2" soln0u

end

# f = Figure(); axf = Axis3(f[1,1]); surface!(axf,Dom0.gridx,Dom0.gridy,soln0u)



# reference_soln, reference_grid = reconstruct_soln(soln0);
refinterp = BicubicInterpolator(Dom0.gridx[:,1],Dom0.gridy[1,:],soln0u)

soln1ref = reconstruct_soln(Dom1,refinterp);
soln2ref = reconstruct_soln(Dom2,refinterp);
soln3ref = reconstruct_soln(Dom3,refinterp);
soln4ref = reconstruct_soln(Dom4,refinterp);
soln5ref = reconstruct_soln(Dom5,refinterp);
soln6ref = reconstruct_soln(Dom6,refinterp);
soln7ref = reconstruct_soln(Dom7,refinterp);
soln8ref = reconstruct_soln(Dom8,refinterp);




relerr = zeros(8);
relerr[1] = norm(soln1.u[2] .- soln1ref)   / norm(soln1ref)
relerr[2] = norm(soln2.u[2] .- soln2ref)   / norm(soln2ref)
relerr[3] = norm(soln3.u[2] .- soln3ref)   / norm(soln3ref)
relerr[4] = norm(soln4.u[2] .- soln4ref)   / norm(soln4ref)
relerr[5] = norm(soln5.u[2] .- soln5ref)   / norm(soln5ref)
relerr[6] = norm(soln6.u[2] .- soln6ref)   / norm(soln6ref)
relerr[7] = norm(soln7.u[2] .- soln7ref)   / norm(soln7ref)
relerr[8] = norm(soln8.u[2] .- soln8ref)   / norm(soln8ref)


grids = [51,101,201,301,401,501,601,701]
conv_rate = log2.(relerr[1:end-1]./relerr[2:end])./log2.(grids[2:end]./grids[1:end-1])




@show relerr
@show conv_rate




# nx = ny = 801
# soln0, Dom0 = sol(nx,ny,order,reference)


# f = Figure(); f1 = Axis3(f[1,1]); surface!(f1,reference_grid.gridx,reference_grid.gridy,reference_soln)
# f = Figure(); f1 = Axis3(f[1,1]); surface!(f1,Dom0.gridx,Dom0.gridy,soln0.u[2])






if save_rates
    df = DataFrame(N=grids,relerr=relerr)
    CSV.write("SingleIslandFieldSelf/SingleIslandField_stretch_O$(order)_k$(k_para)_delta$(δ)_dt4.csv",df)

    trng = collect(0.0:soln1.Δt[2]:1e-2-soln1.Δt[2])
    dftau = DataFrame(t=trng,n7=soln7.τ_hist,n6=soln6.τ_hist,n5=soln5.τ_hist,n4=soln4.τ_hist,n3=soln3.τ_hist,n2=soln2.τ_hist,n1=soln1.τ_hist)
    CSV.write("SingleIslandFieldSelf/SingleIslandField_stretch_tau_O$(order)_k$(k_para)_delta$(δ)_dt4.csv",dftau)
end


if poincare
    println("plotting")
    using GLMakie
    # using CairoMakie
    include("../FaADE_papera/FieldLines.jl")
    poindata = FieldLines.construct_poincare(B,[0.0,1.0],[0,2π])

    g = Figure();

    axg1 = Axis(g[1,1]); 
    scatter!(axg1,poindata.θ[:],poindata.ψ[:],markersize=0.7,color=:black)#,xlims=(0,2π),ylims=(0,1))
    contour!(axg1,soln1.grid.gridy[1,:],soln1.grid.gridx[:,1],soln1.u[2]',levels=100)


    axg2 = Axis(g[1,2]);
    scatter!(axg2,poindata.θ[:],poindata.ψ[:],markersize=0.7,color=:black)#,xlims=(0,2π),ylims=(0,1))
    contour!(axg2,soln2.grid.gridy[1,:],soln2.grid.gridx[:,1],soln2.u[2]',levels=100)
    

    axg3 = Axis(g[2,1]);
    scatter!(axg3,poindata.θ[:],poindata.ψ[:],markersize=0.7,color=:black)#,xlims=(0,2π),ylims=(0,1))
    contour3d!(axg3,soln3.grid.gridy[1,:],soln3.grid.gridx[:,1],soln3.u[2]',levels=100)
end


if plot
    f = Figure(); 
    axf = Axis(f[1,1])

    l1 = scatter!(axf,Dom1.gridx[:,1],soln1.u[2][:,floor(Int,Dom1.nx/2)+1],marker=:+)
    l2 = scatter!(axf,Dom2.gridx[:,1],soln2.u[2][:,floor(Int,Dom2.nx/2)+1],marker=:circle)
    l3 = scatter!(axf,Dom3.gridx[:,1],soln3.u[2][:,floor(Int,Dom3.nx/2)+1],marker=:x)
    l4 = scatter!(axf,Dom4.gridx[:,1],soln4.u[2][:,floor(Int,Dom4.nx/2)+1],marker=:rect)
    l0 = lines!(axf,Dom0.gridx[:,1],soln0u[:,floor(Int,Dom0.nx/2)+1])

    if poincare 
        scatter!(axf,poindata.ψ[:],poindata.θ[:]/2π,color=:black,alpha=0.3,markersize=1)
    end
    Legend(f[1,2],[l1,l2,l3,l4,l0],["51","101","201","401","reference"])



    f1 = Figure();
    axf1 = Axis(f1[1,1])

    l11 = scatter!(axf1,Dom1.gridx[:,1],abs.(soln1.u[2][:,1].-soln1ref[:,1]),marker=:+)
    l12 = scatter!(axf1,Dom2.gridx[:,1],abs.(soln2.u[2][:,1].-soln2ref[:,1]),marker=:circle)
    l13 = scatter!(axf1,Dom3.gridx[:,1],abs.(soln3.u[2][:,1].-soln3ref[:,1]),marker=:x)
    l14 = scatter!(axf1,Dom4.gridx[:,1],abs.(soln4.u[2][:,1].-soln4ref[:,1]),marker=:square)
    
    if poincare 
        scatter!(axf1,poindata.ψ[:],poindata.θ[:]/2π * maximum(abs.(soln4.u[2][:,1].-soln4ref[:,1])),color=:black,alpha=0.3,markersize=1)
    end

    Legend(f1[1,2],[l11,l12,l13,l14],["51","101","201","401"])




    f2 = Figure();
    axf2 = Axis3(f2[1,1])
    surface!(axf2,Dom1.gridx,Dom1.gridy,soln1.u[2])


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

    # Colorbar(f3[1,4],f3scb)
    # Colorbar(f3[2,3],f3scc)
    # Colorbar(f3[2,4],f3scd)

    # cb = Colorbar(f3[1:2,3],)
end







# h = Figure(); axh = Axis(h[1,1]);
# scatter!(axh,poindata.ψ[:],poindata.θ[:],color=:black,alpha=0.7,markersize=1)
# scatter!(axh,Dom0.gridx[:],Dom.gridy[:],markersize=5)

# surface!(axh,soln1.grid.gridx,soln1.grid.gridy,soln0.u[2][1:4:end,1:4:end] .- soln1.u[2])
# surface!(axh,soln2.grid.gridx,soln2.grid.gridy,soln0.u[2][1:8:end,1:8:end] .- soln2.u[2])
# surface!(axh,soln4.grid.gridx[1:10,1:10],soln4.grid.gridy[1:10,1:10],(soln0u[1:16:end,1:16:end] .- soln4.u[2])[1:10,1:10])







# nx = 601
# ny = 601
# Dom = Grid2D(𝒟x,𝒟y,nx,ny)

# # Homogeneous boundary conditions
# BoundaryLeft    = FaADE.SATs.SAT_Dirichlet((y,t) -> 0.0    , Dom.Δx , Left,  order, Dom.Δy, :Cartesian) #x=0
# # BoundaryRight   = FaADE.SATs.SAT_Dirichlet((y,t) -> 1.0    , Dom.Δx , Right, order, Dom.Δy, :Cartesian) #x=1
# BoundaryRight   = FaADE.SATs.SAT_Neumann((y,t) -> 0.0    , Dom.Δx , Right,1, order) #x=1
# BoundaryUp      = FaADE.SATs.SAT_Periodic(Dom.Δy,2,order,Up,    Dom.Δx,:Cartesian)
# BoundaryDown    = FaADE.SATs.SAT_Periodic(Dom.Δy,2,order,Down,  Dom.Δx,:Cartesian)

# BC = FaADE.Inputs.SATBoundaries(BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)

# gdata   = construct_grid(B,Dom,[-2.0π,2.0π],ymode=:period)
# PData   = ParallelData(gdata,Dom,order,κ=k_para)#,B=MagField)

# # Build PDE problem
# Ptmp       = Problem2D(order,u₀,k_perp,k_perp,Dom,BC,S,PData)

# solve(Ptmp,Dom,Δt,1.1Δt,       solver=:theta,  θ=θ)
# soln1 = solve(P1,Dom,0.1Dom.Δx,t_f,  solver=:theta,  θ=θ)

