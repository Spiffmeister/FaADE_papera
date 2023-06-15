#===
    Preservation of asympytotic behaviour

To show we preserve asympytotic behaviour we let κ⟂ → 0
===#

using LinearAlgebra

cd("..")
using Interpolations
push!(LOAD_PATH,"./plas_diff")
push!(LOAD_PATH,"./FaADE")
using FaADE
using plas_diff



# Space Domain

Dx = [0.0,1.0]
Dy = [0.0,1.0]
nx = ny = 21
Dom = Grid2D(Dx,Dy,nx,ny)

# Time domain

Δt = 1.0Dom.Δx^2
t_f = Inf


# Diffusion coefficient

k(x,y) = 0.0


# Initial condition
u₀(x,y) = x

# Boundary conditions
BL  = Boundary(Dirichlet,(y,t) -> 0.0, Left, 1)
BR  = Boundary(Dirichlet,(y,t) -> 0.0, Right, 1)
BUD = BoundaryPeriodic(2)

# Build PDE
P = VariableCoefficientPDE2D(u₀,k,k,2,BL,BR,BUD)



# Parallel component
χₘₙ = 2.1e-3 + 5.0e-3
params = plas_diff.SampleFields.H_params([χₘₙ/2., χₘₙ/3.],[2.0, 3.0],[1.0, 2.0])

function χ_h!(χ,x::Array{Float64},p,t)
    # Hamiltons equations for the field-line Hamiltonian
    # H = ψ²/2 - ∑ₘₙ ϵₘₙ(cos(mθ - nζ))
    χ[1] = x[2] #p_1            qdot        θ
    χ[2] = -sum(p.ϵₘₙ .*(sin.(p.m*x[1] - p.n*t) .* p.m)) #q_1        pdot        ψ
end

gdata = plas_diff.construct_grid(𝒟x,𝒟y,nx,ny,χ_h!,params)



# Solve
soln = solve(P,Dom,Δt,t_f,:cgie)



# Exact solution??



