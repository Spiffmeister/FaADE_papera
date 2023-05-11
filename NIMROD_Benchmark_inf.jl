
using LinearAlgebra
using DelimitedFiles

push!(LOAD_PATH,"../SBP_operators")
using SBP_operators


using GLMakie


Ψ(x,y) = cos(π*x)*cos(π*y)
# order = 2

# Diffusion coefficient
k(x,y) = 0.0


# Domain
𝒟x = [-0.5,0.5]
𝒟y = [-0.5,0.5]

# Homogeneous boundary conditions
BoundaryLeft    = Boundary(Dirichlet,(y,t) -> cos(0.5π)*cos(π*y)    , SBP_operators.Left, 1)
BoundaryRight   = Boundary(Dirichlet,(y,t) -> cos(-0.5π)*cos(π*y)   , SBP_operators.Right, 1)
BoundaryUp      = Boundary(Dirichlet,(x,t) -> cos(π*x)*cos(0.5π)    , SBP_operators.Up, 2)
BoundaryDown    = Boundary(Dirichlet,(x,t) -> cos(π*x)*cos(-0.5π)   , SBP_operators.Down, 2)

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
T(x,y,t) = 2π^2 * t * Ψ(x,y)

N = [17,25,33,41,49,57]

ttol = 1e-6

for order in [2,4]
    pollution = []
    pollnear = []
    rel_error = []
    for n in N
        
        nx = ny = n
        Dom = Grid2D(𝒟x,𝒟y,nx,ny)
        
        # Build PDE problem
        P = VariableCoefficientPDE2D(u₀,k,k,order,BoundaryLeft,BoundaryRight,BoundaryUp,BoundaryDown)

        # Time domain
        Δt = 0.1Dom.Δx^2
        t_f = 1/(2π^2)
        nf = round(t_f/Δt)
        Δt = t_f/nf

        # Parallel penalty
        gdata   = construct_grid(B,Dom,[-2.0π,2.0π],ymode=:stop)
        Pfn = generate_parallel_penalty(gdata,Dom,order,κ=1.0)
        
        # Solve
        soln = solve(P,Dom,Δt,2.1Δt,:cgie,adaptive=false,   nf=nf,source=F,penalty_func=Pfn)
        soln = solve(P,Dom,Δt,t_f,:cgie,adaptive=false,     nf=nf,source=F,penalty_func=Pfn)
        println(nx,"    t_f=",t_f,"    t_f-t=",t_f-soln.t[2],"     Δt=",Δt,"   nf=",nf)

        # Compute the final solution at the final time of the run
        T_exact = zeros(Dom.nx,Dom.ny)
        for j = 1:ny
            for i = 1:nx
                T_exact[i,j] = T(Dom.gridx[i],Dom.gridy[j],soln.t[2]) 
            end
        end

        # push!(pollution, 1/k(0.0,0.0) - soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1])
        push!(pollution, 1.0 - soln.u[2][floor(Int,nx/2)+1,floor(Int,ny/2)+1])
        push!(pollnear,T_exact[floor(Int,nx/2),floor(Int,ny/2)] - soln.u[2][floor(Int,nx/2),floor(Int,ny/2)])
        push!(rel_error, norm(T_exact .- soln.u[2])/norm(T_exact))

        # P = Figure(resolution=(1200,1200))
        # Pa = Axis3(P[1,1])
        # wireframe!(Pa,Dom.gridx,Dom.gridy,abs.(soln.u[2].-T_exact))
        # name=string("kinf_O",order,"_n",n,".png")
        # save(name,P)

        # P2 = Figure(resolution=(1200,1200))
        # Pa2 = Axis(P2[1,1])
        # # Pa21 = lines!(Pa2,soln.u[2][floor(Int,nx/2)+1,:])
        # Pa22 = lines!(Pa2,abs.(T_exact[floor(Int,nx/2)+1,:].-soln.u[2][floor(Int,nx/2)+1,:]))
        # # axislegend(Pa2,[Pa21,Pa22],["numerical","exact"])
        # name=string("aaa_lines/kinf_O",order,"_n",n,".png")
        # save(name,P2)

    end
    # nameappend=string("k=",k(0,0))

    open(string("limit/NB_limit_pollution_O",order,".csv"),"w") do io
        writedlm(io,[N pollution pollnear])
    end

    open(string("limit/NB_limit_relerr_O",order,".csv"),"w") do io
        writedlm(io,[N rel_error])
    end

    println("pollution=",pollution)
    println("poll near",pollnear)
    println("rel error=",rel_error)
end



#=
N = 1:10
levellist = [T(Dom.gridx[i],Dom.gridy[i],t_f) for i in N]

contour(Dom.gridx,Dom.gridy,T_exact,
    levels=levellist)



grid_contour = LinRange(-0.5,0.5,100);
T_contour = zeros(100,100);
for i = 1:100, j = 1:100
    T_contour[i,j] = T(grid_contour[i],grid_contour[j],t_f);
end

contour(grid_contour,grid_contour,T_contour,
    levels=[T(Dom.gridx[1],Dom.gridy[1],t_f)],legend=false)
for i = 1:20
    pt = rand(1:33,2)
    contour!(grid_contour,grid_contour,T_contour,levels=[T(Dom.gridx[pt[1]],Dom.gridy[pt[2]],t_f)])
    plot!([Dom.gridx[pt[1]],gdata.Fplane.x[pt[1],pt[2]]],[Dom.gridy[pt[2]],gdata.Fplane.y[pt[1],pt[2]]])
end

contour!(Dom.gridx,Dom.gridy,T_exact,levels=[T(Dom.gridx[18],Dom.gridy[18],t_f)])
plot!([Dom.gridx[18],gdata.Fplane.x[18,18]],[Dom.gridy[18],gdata.Fplane.y[18,18]])




# scatter!([Dom.gridx[5],gdata.Fplane.x[5,17]],[Dom.gridy[17],gdata.Fplane.y[5,17]])

# Solve

# GLMakie.surface(soln.u[2])

# println(T(0.0,0.0,t_f) .- soln.u[2][argmin(abs.(Dom.gridx)),argmin(abs.(Dom.gridy))])

# println(soln.u[2][argmin(abs.(Dom.gridx)),argmin(abs.(Dom.gridy))] .- k(0.0,0.0))

# using Plots
# surface(T_exact)
# surface(soln.u[2])

# ϵ = zeros(nruns)
# for i = 1:nruns
#     ϵ[i] = norm( 1/k(0,0) - solns[i].soln.u[2][ argmin(abs.(solns[i].Dom.gridx)), argmin(abs.(solns[i].Dom.gridy)) ] )
# end

=#
