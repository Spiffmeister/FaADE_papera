using LinearAlgebra
using DelimitedFiles

push!(LOAD_PATH,"../SBP_operators")
using SBP_operators





Ψ(x,y) = cos(π*x)*cos(π*y)


# Domain
𝒟x = [-0.5,0.5]
𝒟y = [-0.5,0.5]

# Homogeneous boundary conditions
BoundaryLeft    = Boundary(Dirichlet,(y,t) -> cos(0.5π)*cos(π*y)    , Left, 1)
BoundaryRight   = Boundary(Dirichlet,(y,t) -> cos(-0.5π)*cos(π*y)   , Right, 1)
BoundaryUp      = Boundary(Dirichlet,(x,t) -> cos(π*x)*cos(0.5π)    , Up, 2)
BoundaryDown    = Boundary(Dirichlet,(x,t) -> cos(π*x)*cos(-0.5π)   , Down, 2)

BoundaryLeftRight = PeriodicBoundary(1)
BoundaryUpDown = PeriodicBoundary(2)

# Initial condition
u₀(x,y) = 0.0
# Source term
F(x,y,t) = 2π^2*cos(π*x)*cos(π*y)
# Magnetic field
function B(X,x,p,t)
    X[1] = -π*cos(π*x[1])*sin(π*x[2])
    X[2] = π*sin(π*x[1])*cos(π*x[2])
    # X[3] = 0.0
end
# Exact solution


N = [17,25,33,41,49,57]

for k_perp = [1.0,1e-3,1e-6,1e-9]
    k(x,y) = k_perp
    T(x,y,t) = (1.0 - exp(-2.0*k_perp*π^2*t) )/( k_perp )*Ψ(x,y)
    # Diffusion coefficient
    for order in [2,4]
        rel_error = []
        abs_error = []
        
        P = VariableCoefficientPDE2D(u₀,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)
        # P = VariableCoefficientPDE2D(u₀,k,k,order,BoundaryLeftRight,BoundaryUpDown)

        println("===",k_perp,"===",order,"===")
        for n in N
            nx = ny = n
            Dom = Grid2D(𝒟x,𝒟y,nx,ny)
            
            # Build PDE problem

            # Time domain
            Δt = 0.01Dom.Δx^2
            t_f = 1/(2π^2)
            nf = round(t_f/Δt)
            Δt = t_f/nf

            gdata   = construct_grid(B,Dom,[-2.0π,2.0π],ymode=:stop)
            Pfn = generate_parallel_penalty(gdata,Dom,order) # Generate a parallel penalty with a modified penalty parameter

            #remap all points back to themselves
            for i = 1:n 
                gdata.Fplane.x[:,i] = Dom.gridx
                gdata.Fplane.y[i,:] = Dom.gridy
                gdata.Bplane.x[:,i] = Dom.gridx
                gdata.Bplane.y[i,:] = Dom.gridy
            end



            soln = solve(P,Dom,Δt,2.1Δt,:cgie,adaptive=false   ,nf=nf,source=F,penalty_func=Pfn)
            soln = solve(P,Dom,Δt,t_f,:cgie,adaptive=false     ,nf=nf,source=F,penalty_func=Pfn)
            println(nx,"    t_f=",t_f,"    t_f-t=",t_f-soln.t[2],"     Δt=",Δt,"   nf=",nf)

            T_exact = zeros(Dom.nx,Dom.ny)
            for j = 1:ny
                for i = 1:nx
                    T_exact[i,j] = T(Dom.gridx[i],Dom.gridy[j],soln.t[end])
                end
            end
            

            push!(rel_error, norm(T_exact .- soln.u[2])/norm(T_exact))
            push!(abs_error, norm(T_exact .- soln.u[2]))

            # p = surface(soln.u[2],dpi=600)
            # pltname = string("perp_identityNB_kperp_",k_perp,"_pollution_O",order,".png")
            # savefig(p,pltname)

        end
        nameappend=string("k=",k(0,0))

        open(string("perp_identity/NB_kperp_",k_perp,"_relerr_O",order,".csv"),"w") do io
            writedlm(io,[N rel_error abs_error])
        end

        println("rel error=",rel_error)
    end
end

# Solve




# println(T(0.0,0.0,t_f) .- soln.u[2][argmin(abs.(Dom.gridx)),argmin(abs.(Dom.gridy))])

# println(soln.u[2][argmin(abs.(Dom.gridx)),argmin(abs.(Dom.gridy))] .- k(0.0,0.0))


# using Plots
# surface(T_exact)
# surface(soln.u[2])


# ϵ = zeros(nruns)
# for i = 1:nruns
#     ϵ[i] = norm( 1/k(0,0) - solns[i].soln.u[2][ argmin(abs.(solns[i].Dom.gridx)), argmin(abs.(solns[i].Dom.gridy)) ] )
# end


