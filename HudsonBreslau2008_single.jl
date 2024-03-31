"""
Section 4.4 - Generating field from the Hudson-Breslau 2008 paper
"""
using LinearAlgebra
using Revise
using FaADE

using BasicInterpolators
using JLD2

plot = true
poincare = false
contour = false



###
𝒟x = [0.0,1.0]
𝒟y = [-π,π]
nx = 1001
ny = 601

function GPack(x,nx;B=0.5)
    α = (10/(nx-1))^2
    return B + α * sinh( asinh((1-B)/α)*x + asinh((0-B)/α)*(1-x) )
end
Dx, Dy = FaADE.Grid.meshgrid(GPack.(LinRange(𝒟x[1],𝒟x[2],nx),nx,B=0.6),collect(LinRange(𝒟y[1],𝒟y[2],ny)))
Dom = Grid2D(Dx,Dy,ymap=false)
# Dom = Grid2D(Dx,Dy,nx,ny)



θ = 0.5
order = 2

k_para = 1.0e10
k_perp = 1.0


# Time setup
Δt = 1e-5
t_f = 1e-1
# target=1e-4



ϵ = 2.1e-3 #Perturbation parameter
params = (ϵₘₙ = [ϵ/2., ϵ/3.], m=[2.0, 3.0], n=[1.0, 2.0])
function B(X,x::Array{Float64},p,t)
    X[2] = x[1] #p_1            qdot        θ
    X[1] = -sum(p.ϵₘₙ .*(sin.(p.m*x[2] - p.n*t) .* p.m)) #q_1        pdot        ψ
end
dH(X,x,p,t) = B(X,x,params,t)
gdata   = construct_grid(dH,Dom,[-2.0π,2.0π],ymode=:period)
PData   = FaADE.ParallelData(gdata,Dom,order,κ=k_para)



u₀(x,y) = x



BoundaryLeft    = FaADE.SATs.SAT_Dirichlet((y,t) -> 0.0, Dom.Δx, Left ,order,Dom.Δy,:Cartesian)
BoundaryRight   = FaADE.SATs.SAT_Dirichlet((y,t) -> 1.0, Dom.Δx, Right,order,Dom.Δy,:Cartesian)
BoundaryUp      = FaADE.SATs.SAT_Periodic(Dom.Δy,2,order,Up,    Dom.Δx,:Cartesian)
BoundaryDown    = FaADE.SATs.SAT_Periodic(Dom.Δy,2,order,Down,  Dom.Δx,:Cartesian)

BC = FaADE.Inputs.SATBoundaries(BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)



# Build PDE problem
P = Problem2D(order,u₀,k_perp,k_perp,Dom,BC,nothing,PData)



solve(P,Dom,Δt,1.1Δt,solver=:theta,  θ=θ)#, adaptive=true)
soln = solve(P,Dom,Δt,t_f,  solver=:theta,  θ=θ)#, adaptive=true, target=target)
solnu = soln.u[2]
# save("HB/HudsonBreslau2008$(nx)$(ny).jld2","solnu",solnu)
# @load "HB/HudsonBreslau20081001601.jld2" solnu

if plot
    println("plotting")
    using GLMakie
    # using CairoMakie

    f = Figure(); 
    ax_f = Axis3(f[1,1]);
    # ax2_f = Axis(f[1,2]);
    # surface!(ax_f,Dom.gridx,Dom.gridy,soln.u[2])
    surface!(ax_f,Dom.gridx,Dom.gridy,solnu)
    # lines!(ax2_f,Dom.gridx[:,1],soln.u[2][:,floor(Int,Dom.nx/2)+1])
    # wireframe!(ax,Dom.gridx,Dom.gridy,soln.u[2])
    # contour3d!(ax,soln.u[2],levels=100)

    g = Figure(); 
    ax_g = Axis(g[1,1]); 
    contour3d!(ax_g,Dom.gridy[1,:],Dom.gridx[:,1],soln.u[2]',levels=100)
    Colorbar(g[1,2])
end


if poincare
    # include("../FaADE_papera/FieldLines.jl")
    include("../paper_JCP2023/FieldLines.jl")


    poindata = FieldLines.construct_poincare(dH,[0.0,1.0],[-π,π],N_trajs=400,N_orbs=400)



    xlo = 0.45; xup = 0.72;
    xrng = Dom.gridx[xlo .≤ Dom.gridx[:,1] .≤ xup,1];

    solnutrans = solnu;
    solnutrans = solnutrans[xlo .≤ Dom.gridx[:,1] .≤ xup,:];
    θrng = poindata.θ[xlo .≤ poindata.ψ .≤ xup];
    ψrng = poindata.ψ[xlo .≤ poindata.ψ .≤ xup];



    h = Figure(size=(1400,600), fontsize=30);
    hg = h[1,1] = GridLayout();

    ax_h = Axis(hg[1,1], xlabel=L"\theta", ylabel=L"\psi",xticks=([-π/2,0,π/2],[L"-\pi/2",L"0",L"\pi/2"]));

    scatter!(ax_h,poindata.θ[:],poindata.ψ[:],markersize=1.0,color=(:black,0.3))
        
    hca = GLMakie.contour!(ax_h,Dom.gridy[1,:],Dom.gridx[:,1],solnu',levels=0.0:0.025:1.0,linewidth=2.5)

    xlims!(ax_h,-π,π)
    ylims!(ax_h,0.0,1.0)

    poly!(ax_h,Point2f[(-π+1e-2,xlo),(π-1e-2,xlo),(π-1e-2,xup),(-π+1e-2,xup)],strokewidth=5,strokecolor=:blue,color=(:white,0.0))


    # ax_h2 = Axis(hg[1,2], xlabel=L"\theta", yaxisposition=:right,xticks=([-π/2,0,π/2],[L"-\pi/2",L"0",L"\pi/2"]));
    ax_h2 = Axis(hg[1,2],xticks=([-π/2,0,π/2],["","",""]));
    hidedecorations!(ax_h2)

    scatter!(ax_h2,θrng,ψrng,markersize=1.0,color=(:black,0.5))

    hca2 = GLMakie.contour!(ax_h2,Dom.gridy[1,:],xrng,solnutrans',levels=0.0:0.025:1.0,linewidth=2.5,colorrange=hca.colorrange)

    xlims!(ax_h2,-π,π)
    ylims!(ax_h2,xlo,xup)

    pr = poly!(ax_h2,Point2f[(-π+1e-2,xlo),(π-1e-2,xlo),(π-1e-2,xup),(-π+1e-2,xup)],strokewidth=5,strokecolor=:blue,color=(:white,0.0))
    axislegend(ax_h2,[pr],["zoomed region"])


    colgap!(hg,10)



    import Contour as Cont



    Itp = BicubicInterpolator(Dom.gridx[:,1],Dom.gridy[1,:],solnu)

    c1 = Cont.contour(Dom.gridx,Dom.gridy,solnu,Itp(0.505,-π))
    c1x = [x[1] for x in c1.lines[1].vertices];
    c1y = [x[2] for x in c1.lines[1].vertices];

    c2 = Cont.contour(Dom.gridx,Dom.gridy,solnu,Itp(0.675,0.0))
    c2x = [x[1] for x in c2.lines[1].vertices];
    c2y = [x[2] for x in c2.lines[1].vertices];

    Pcon21 = lines!(ax_h,c1y,c1x,color=:red)
    Pcon22 = lines!(ax_h,c2y,c2x,color=:red)

    Pcon23 = lines!(ax_h2,c1y,c1x,color=:red)
    Pcon24 = lines!(ax_h2,c2y,c2x,color=:red)

    axislegend(ax_h,[Pcon21],["O point contour"])

    
    save("HB/HudsonBreslau2008Contour.png",h,px_per_unit=4.0)

end

