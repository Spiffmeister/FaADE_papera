#=
    Example for computing in a slab case
=#
# push!(LOAD_PATH,"../plas_diff/")
# using plas_diff
include("FieldLines.jl")




Dx = [0.0,1.0]
Dy = [-π,π]
ϵ = 2.1e-3 + 5e-3 #Perturbation parameter
p = (ϵₘₙ = [ϵ/2., ϵ/3.], m=[2.0, 3.0], n=[1.0, 2.0])
function χ_h!(χ,x::Array{Float64},params,t)
    # Hamiltons equations for the field-line Hamiltonian
    # H = ψ²/2 - ψ(ψ-1) ∑ₘₙ ϵₘₙ(cos(mθ - nζ)) 
    χ[1] = x[2] #p_1            qdot        θ
    χ[2] = -sum(p.ϵₘₙ .*(sin.(p.m*x[1] - p.n*t) .* p.m)) #q_1        pdot        ψ
end



function f_1!(ψ,θ,params,t)
    ψ = θ
end
function f_2!(ψ,θ,params,t)
    -sum(p.ϵₘₙ .* sin.(p.m*ψ - p.n*t) .* p.m)
end


pdata = FieldLines.symp_poincare(f_1!,f_2!,Dx,Dy,N_trajs=1000,N_orbs=400)

ptrace1 = FieldLines.tracer(χ_h!,100,[1.0,0.3])
ptrace2 = FieldLines.tracer(χ_h!,100,[0.5,0.6])
ptrace3 = FieldLines.tracer(χ_h!,100,[0.2,0.8])


# using CairoMakie
using GLMakie


F = Figure(resolution=(1600,1200),fontsize=40)
# F = Figure()

Ax = Axis(F[1,1],xlabel=L"\theta",ylabel=L"\psi",xlabelsize=50,ylabelsize=50)

P = scatter!(Ax,pdata.θ[:],pdata.ψ[:],markersize=3.0,color=:black)

P1 = scatter!(Ax,ptrace1[1],ptrace1[2],markersize=25.0)
P2 = scatter!(Ax,ptrace2[1],ptrace2[2],markersize=25.0)
P3 = scatter!(Ax,ptrace3[1],ptrace3[2],markersize=25.0)

xlims!(-π,π)
ylims!(0,1)
# Ax.yticks = ()
Ax.xticks = (-π:π/2:π,["π", "-π/2", "0", "π/2", "π"])


axislegend(Ax,[P1,P2,P3],["(-1.0,0.3)","(0.5,6.0)","(0.2,0.8)"],"Initial Condition",position=:rb,labelsize=30)

# save("PoincareSection.pdf",F)

