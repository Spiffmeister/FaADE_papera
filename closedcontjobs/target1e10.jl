using LinearAlgebra

push!(LOAD_PATH,"../SBP_operators")
push!(LOAD_PATH,"../plas_diff")
using SBP_operators
using plas_diff




###
Dx = [0.0,1.0]
Dy = [-π,π]
nx = 51
ny = 51
Dom = Grid2D(Dx,Dy,nx,ny)


order = 4
target = 1e-10

Δt = 0.1Dom.Δx^2
# t_f = 100.0
t_f = Inf

println("Parallel grid construction")
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
@time PGrid = SBP_operators.construct_grid(dH,Dom,[-2π,2π])
Pfn = SBP_operators.generate_parallel_penalty(PGrid,Dom,order)


u0(x,y) = x

BoundaryLeft = Boundary(Dirichlet,(y,t) -> 0.0,SBP_operators.Left,1)
BoundaryRight = Boundary(Dirichlet,(y,t) -> 1.0,SBP_operators.Right,1)
BoundaryUpDown = PeriodicBoundary(2)


println("Poincare construction")
params = plas_diff.SampleFields.H_params([ϵ/2., ϵ/3.], [2.0, 3.0], [1.0, 2.0])
function χ_h!(χ,x::Array{Float64},p,t)
    # Hamiltons equations for the field-line Hamiltonian
    # H = ψ²/2 - ∑ₘₙ ϵₘₙ(cos(mθ - nζ))
    χ[1] = x[2] #p_1            qdot        θ
    χ[2] = -sum(p.ϵₘₙ .*(sin.(p.m*x[1] - p.n*t) .* p.m)) #q_1        pdot        ψ
end
@time pdata = plas_diff.poincare(χ_h!,params,N_trajs=750,N_orbs=200,x=Dx,y=Dy)


println("Begin compile solve")
k(x,y) = 1.0
P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)
solve(P,Dom,Δt,2.1Δt,:cgie,adaptive=true,penalty_func=Pfn,target=target)

println("Begin solve")
k(x,y) = 1.0e-6
P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)
@time soln1   = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn,target=target)




println("Plotting")

using GLMakie
using CairoMakie
using Contour
using Interpolations


xvals = LinRange(0.0,1.0,101);

function findcontours(xvals,soln,Dom)
    itp = LinearInterpolation(Dom.gridx,soln.u[2][:,26])
    uvals = itp(xvals)
    return uvals
end


c1  = findcontours(xvals,soln1,Dom);


pmode = CairoMakie



F =     pmode.Figure(resolution=(1600,1200),fontsize=40)
Ax =    Axis(F[1,1],xlabel=L"\theta",ylabel=L"\psi",xlabelsize=50,ylabelsize=50)
P =     scatter!(Ax,pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)
ylims!(0.0,1.0)
xlims!(-π,π)
Pcon =  contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=0.05:0.05:0.95,linewidth=4.0)
Colorbar(F[1,2],Pcon,label=L"u",labelsize=50)

Pcon2 =  contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=findcontours([0.504,0.67314],soln1,Dom),linewidth=3.0,color=:red)

axislegend(Ax,[Pcon2],["O point contour"])

# name = string("./FieldFeatures_nasty.pdf")
# pmode.save(name, F)#, resolution=(1600,1200), transparency=true)





#=
using DelimitedFiles

open(string("FieldFeatures_poincare.csv"),"w") do io
    writedlm(io,[pdata.θ[:] pdata.ψ[:]])
end



open(string("FieldFeatures_temp.csv"),"w") do io
    writedlm(io,soln1.u[2])
end


using GLMakie
GLMakie.activate!()

GLMakie.surface(Dom.gridx*pi,Dom.gridy,soln1.u[2])
GLMakie.wireframe!(Dom.gridx*pi,Dom.gridy,soln1.u[2],color=(:black,0.3),transparancy=true)



GLMakie.lines(Dom.gridx,soln1.u[2][:,51])






F =     GLMakie.Figure()
Ax =    Axis(F[1,1],xlabel=L"\theta",ylabel=L"\psi",xlabelsize=50,ylabelsize=50,title=string("Δu=",target))
P =     scatter!(Ax,pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)
ylims!(0.0,1.0)
xlims!(-π,π)
Pcon =  contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=0.05:0.05:0.95,linewidth=4.0)
Colorbar(F[1,2],Pcon,label=L"u",labelsize=50)


Pcon2 =  contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=findcontours([0.504,0.67314],soln1,Dom),linewidth=3.0,color=:red)

axislegend(Ax,[Pcon2],["O point contour"])

name = string("./FieldFeatures.pdf")
pmode.save(name, F)#, resolution=(1600,1200), transparency=true)


contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[0.6035],linewidth=3.0,color=:red)


contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=findcontours([x for x in 0.666:0.01:0.71],soln1,Dom),linewidth=3.0,color=:red)
contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=findcontours([x for x in 0.67:0.001:0.69],soln1,Dom),linewidth=3.0,color=:red)


contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.49,soln1,Dom)],linewidth=3.0,color=:red)
contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.51,soln1,Dom)],linewidth=3.0,color=:red)
contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.62,soln1,Dom)],linewidth=3.0,color=:red)
contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.625,soln1,Dom)],linewidth=3.0,color=:red)
contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.65,soln1,Dom)],linewidth=3.0,color=:red)
contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.67,soln1,Dom)],linewidth=3.0,color=:red)
contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.68,soln1,Dom)],linewidth=3.0,color=:red)
contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.69,soln1,Dom)],linewidth=3.0,color=:red)

contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.6809,soln1,Dom)],linewidth=3.0,color=:red)
contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.6801,soln1,Dom)],linewidth=3.0,color=:red)

contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=findcontours([x for x in 0.68:0.000001:0.681],soln1,Dom),linewidth=3.0,color=:red)



contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[0.5812944722968906],linewidth=3.0,color=:red)

contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[0.58112213133933],linewidth=3.0,color=:red)



contour!(Ax,Dom.gridy,Dom.gridx,soln1.u[2]',levels=[findcontours(0.69,soln1,Dom)],linewidth=3.0,color=:red)


using JLD2
jldsave()


=#

