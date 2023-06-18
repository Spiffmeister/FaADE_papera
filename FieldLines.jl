


module FieldLines
    using DifferentialEquations



    struct poin_data
        ψ :: Vector{Float64}
        θ :: Vector{Float64}
    end

    """
        construct_grid(χ::Function,grid::Grid2D,z::Vector)
    Constructs the backwards and forward planes for a given plane

    Inputs:
    - Field line ODE that returns ``[x,y]``
    - GridType
    - z values of planes to trace to

    Outputs:
    - ParallelGrid object (see [ParallelGrid](@ref))
    """
    function construct_poincare(H::Function,x::Vector{T},y::Vector{T};N_trajs::Int=500,N_orbs::Int=100) where T

        ζ = (0.0,2*N_orbs*π)
        ζ₀ = π*collect(0:2:2*N_orbs)

        x₀ = rand(2,N_trajs)

        # Ensure things are within domains
        x₀[1,:] = (x[end] - x[1])*x₀[1,:] .+ x[1] #ψ
        x₀[2,:] = (y[end] - y[1])*x₀[2,:] .+ y[1] #θ

        function prob_fn(prob,i,repeat)
            remake(prob,u0=x₀[:,i])
        end

        # Construct the problem and solve the trajectories in parallel
        P = ODEProblem(H,x₀[:,1],ζ)
        EP = EnsembleProblem(P,prob_func=prob_fn)
        sim = solve(EP,EnsembleDistributed(),trajectories=N_trajs,tstops=ζ₀)

        # Loop though outputs and store plane intersecetions
        ind = zeros(Int64,length(ζ₀))
        N_orbs += 1
        data = zeros(2,N_trajs*N_orbs)
        for i = 1:N_trajs
            t = sim.u[i].t
            for j = 1:length(ζ₀)
                ind[j] = argmin(abs.(t .- 2π*j))[1]
            end
            u = sim.u[i].u[ind]
            data[:,(i-1)*N_orbs+1:i*N_orbs] = mod.(hcat(u...),2π)
        end

        if y[1] == -π
            data[2,:] = rem2pi.(data[2,:],RoundNearest)
        end

        return poin_data(data[1,:],data[2,:])
    end


    function symp_poincare(f1::Function,f2::Function,x::Vector{T},y::Vector{T};N_trajs::Int=500,N_orbs::Int=100) where T

        ζ = (0.0,2*N_orbs*π)
        ζ₀ = π*collect(0:2:2*N_orbs)

        x₀ = rand(2,N_trajs)

        # Ensure things are within domains
        x₀[1,:] = (x[end] - x[1])*x₀[1,:] .+ x[1] #ψ
        x₀[2,:] = (y[end] - y[1])*x₀[2,:] .+ y[1] #θ

        function prob_fn(prob,i,repeat)
            remake(prob,u0=ArrayPartition(x₀[2,i],x₀[1,i]))
        end

        # Construct the problem and solve the trajectories in parallel
        P = DynamicalODEProblem(f1,f2,x₀[1,1],x₀[1,2],ζ)
        EP = EnsembleProblem(P,prob_func=prob_fn)
        sim = solve(EP,KahanLi8(),EnsembleDistributed(),trajectories=N_trajs,tstops=ζ₀)

        # Loop though outputs and store plane intersecetions
        ind = zeros(Int64,length(ζ₀))
        N_orbs += 1
        datax = zeros(N_trajs*N_orbs)
        datay = zeros(N_trajs*N_orbs)

        for i = 1:N_trajs
            u = zeros(2,length(sim.u[i].u))
            for j = 1:length(sim.u[i].u)
                u[1,j] = sim.u[i].u[j][1]
                u[2,j] = sim.u[i].u[j][2]
            end
            datay[(i-1)*N_orbs+1:i*N_orbs] = u[1,:]
            datax[(i-1)*N_orbs+1:i*N_orbs] = u[2,:]
        end
# println(datay[1:10])
        if y[1] == -π
            datax = rem2pi.(datax,RoundNearest)
            datay = rem2pi.(datay,RoundNearest)
        end

        return poin_data(datax,datay)
    end

    function tracer(H::Function,N_orbs::Int64,x₀::Array{Float64};dir=1.0)
        ζ = (0.0,dir*2*N_orbs*pi)
        ζ₀ = pi*collect(0:dir*2:2*N_orbs)
        P = ODEProblem(H,x₀,ζ)
        sol = solve(P,tstops=ζ₀)

        u3_ind = zeros(Int,length(ζ₀))
        for n = 1:length(ζ₀)
            u3_ind[n] = argmin(abs.(sol.t .- 2*n*pi))[1]
        end

        θ = [rem2pi(x[1],RoundNearest) for x in sol.u[u3_ind]]
        ψ = [x[2] for x in sol.u[u3_ind]]

        return θ,ψ

    end




end


