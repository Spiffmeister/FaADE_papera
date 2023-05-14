using LinearAlgebra

push!(LOAD_PATH,"../SBP_operators")
push!(LOAD_PATH,"../plas_diff")
using SBP_operators
using plas_diff




###
Dx = [0.0,1.0]
Dy = [-π,π]
nx = 21
ny = 21
Dom = Grid2D(Dx,Dy,nx,ny)


order = 2
k(x,y) = 1.0e-6

Δt = 0.1Dom.Δx^2
t_f = 100.0

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


P = VariableCoefficientPDE2D(u0,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUpDown)

soln = solve(P,Dom,Δt,t_f,:cgie,adaptive=true,penalty_func=Pfn)




params = plas_diff.SampleFields.H_params([ϵ/2., ϵ/3.], [2.0, 3.0], [1.0, 2.0])
function χ_h!(χ,x::Array{Float64},p,t)
    # Hamiltons equations for the field-line Hamiltonian
    # H = ψ²/2 - ∑ₘₙ ϵₘₙ(cos(mθ - nζ))
    χ[1] = x[2] #p_1            qdot        θ
    χ[2] = -sum(p.ϵₘₙ .*(sin.(p.m*x[1] - p.n*t) .* p.m)) #q_1        pdot        ψ
end
pdata = plas_diff.poincare(χ_h!,params,N_trajs=500,N_orbs=200,x=Dx,y=Dy)




aaa = plas_diff.construct_grid(Dx,Dy,nx,ny,χ_h!,params)










using GLMakie




F = Figure()

Ax = Axis(F[1,1],xlabel=L"\theta",ylabel=L"\psi")

scatter!(Ax,pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)

con = contour!(Ax,Dom.gridy,Dom.gridx,soln.u[2]',levels=10,linewidth=4.0)
# con = contourf!(Ax,Dom.gridy,Dom.gridx,soln.u[2]')

xlims!(-π,π)
ylims!(0,1)

Ax.xticks = (-π:π/2:π,["π", "-π/2", "0", "π/2", "π"])
Ax.yticks = (0.0:0.1:1.0)
Colorbar(F[1,2],con,label=L"u")




# contourf(Dom.gridy,Dom.gridx,soln.u[2]')



# save("./FieldFeatures.pdf", p3)#, resolution=(1600,1200), transparency=true)

