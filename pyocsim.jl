using LinearAlgebra

cd("..")
using Interpolations
push!(LOAD_PATH,"./SBP_operators")
using SBP_operators





###
𝒟x = [0.0,1.0]
𝒟y = [-π,π]
nx = 21
ny = 21
Dom = Grid2D(𝒟x,𝒟y,nx,ny)



# params = plas_diff.SampleFields.H_params([0.],[0.],[0.])
χₘₙ = 2.1e-3 + 5.0e-3
params = (ϵₘₙ=[χₘₙ/2., χₘₙ/3.],m=[2.0, 3.0],n=[1.0, 2.0])
function χ_h!(χ,x::Array{Float64},p,t)
    # Hamiltons equations for the field-line Hamiltonian
    # H = ψ²/2 - ∑ₘₙ ϵₘₙ(cos(mθ - nζ))
    χ[2] = x[1] #p_1            qdot        θ
    χ[1] = -sum(p.ϵₘₙ .*(sin.(p.m*x[2] - p.n*t) .* p.m)) #q_1        pdot        ψ
    # χ[1] = x[2] #p_1            qdot        θ
    # χ[2] = -sum(p.ϵₘₙ .*(sin.(p.m*x[1] - p.n*t) .* p.m)) #q_1        pdot        ψ
end

χ(χ,x,p,t) = χ_h!(χ,x,params,t)


Pgrid   = construct_grid(χ,Dom,[-2π,2π],ymode=:stop)

pfn     = generate_parallel_penalty(Pgrid,Dom,2)


kx(x,y) = 1.0
ky(x,y) = 1.0

u₀(x,y) = x

BoundaryLeft    = Boundary(Dirichlet,(y,t) -> 0.0,Left,1)
BoundaryRight   = Boundary(Dirichlet,(y,t) -> 0.0,Right,1)
BoundaryUpDown  = PeriodicBoundary(2)

P = VariableCoefficientPDE2D(u₀,kx,ky,2,BoundaryLeft,BoundaryRight,BoundaryUpDown)


t_f = 1.0
Δt = Dom.Δy^2/100

soln = solve(P,Dom,Δt,5.1Δt,:cgie,penalty_func=pfn)


#=
χₘₙ = 2.1e-3 + 5.0e-3
params = plas_diff.SampleFields.H_params([χₘₙ/2., χₘₙ/3.],[2.0, 3.0],[1.0, 2.0])

function χ_h!(χ,x::Array{Float64},p,t)
    # Hamiltons equations for the field-line Hamiltonian
    # H = ψ²/2 - ∑ₘₙ ϵₘₙ(cos(mθ - nζ))
    χ[1] = x[2] #p_1            qdot        θ
    χ[2] = -sum(p.ϵₘₙ .*(sin.(p.m*x[1] - p.n*t) .* p.m)) #q_1        pdot        ψ
end

gdata = plas_diff.construct_grid(𝒟x,𝒟y,nx,ny,χ_h!,params)


gdata.z_planes[1].x
# gdata.z_planes[1].y
Pgrid.Fplane
=#
