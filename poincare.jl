#=
    Example for computing in a slab case
=#
push!(LOAD_PATH,"../plas_diff/")
using plas_diff




Dx = [0.0,1.0]
Dy = [-π,π]
k = 2.1e-3 + 5e-3 #Perturbation parameter
params = plas_diff.SampleFields.H_params([k/2., k/3.], [2.0, 3.0], [1.0, 2.0])
function χ_h!(χ,x::Array{Float64},p,t)
    # Hamiltons equations for the field-line Hamiltonian
    # H = ψ²/2 - ψ(ψ-1) ∑ₘₙ ϵₘₙ(cos(mθ - nζ)) 
    χ[1] = x[2] #p_1            qdot        θ
    χ[2] = -sum(p.ϵₘₙ .*(sin.(p.m*x[1] - p.n*t) .* p.m)) #q_1        pdot        ψ
end



pdata = plas_diff.poincare(χ_h!,params,N_trajs=500,N_orbs=200,x=Dx,y=Dy)

ptrace1 = plas_diff.tracer(χ_h!,params,100,[1.0,0.3])
ptrace2 = plas_diff.tracer(χ_h!,params,100,[0.5,0.6])
ptrace3 = plas_diff.tracer(χ_h!,params,100,[0.2,0.8])


using CairoMakie
# using GLMakie


F = Figure(resolution=(1600,1200),fontsize=20)
# F = Figure()

Ax = Axis(F[1,1],xlabel=L"\theta",ylabel=L"\psi")

P = scatter!(Ax,pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)
P1 = scatter!(Ax,ptrace1[1],ptrace1[2],markersize=25.0)
P2 = scatter!(Ax,ptrace2[1],ptrace2[2],markersize=25.0)
P3 = scatter!(Ax,ptrace3[1],ptrace3[2],markersize=25.0)

xlims!(-π,π)
ylims!(0,1)
# Ax.yticks = ()
Ax.xticks = (-π:π/2:π,["π", "-π/2", "0", "π/2", "π"])


axislegend(Ax,[P1,P2,P3],["(-1.0,0.3)","(0.5,6.0)","(0.2,0.8)"],"Initial Condition",position=:rb,labelsize=30)

save("PoincareSection.pdf",F)

