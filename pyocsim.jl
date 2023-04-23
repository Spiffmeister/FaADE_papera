using LinearAlgebra

cd("..")
using Interpolations
push!(LOAD_PATH,"./plas_diff")
push!(LOAD_PATH,"./SBP_operators")
using SBP_operators
using plas_diff





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
    χ[1] = x[2] #p_1            qdot        θ
    χ[2] = -sum(p.ϵₘₙ .*(sin.(p.m*x[1] - p.n*t) .* p.m)) #q_1        pdot        ψ
end

χ(χ,x,t) = χ_h(χ,x,params,t)


SBP_operators.Parallel.construct_grid(χ,Dom,[-2π,2π])



