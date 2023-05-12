


# using GLMakie
using CairoMakie


B(x) = Point2f(-π*cos(π*x[1])*sin(π*x[2]),
    π*sin(π*x[1])*cos(π*x[2])
    )


ψ(x,y) = cos(π*x)*cos(π*y)

XY = LinRange(-0.5,0.5,100)
Z = [ψ(x,y) for x in XY, y in XY]



p = Figure(resolution=(1600,1200),fontsize=40)
# p = Figure()

ax = Axis(p[1,1], xlabel="x",ylabel="y", xlabelsize=50,ylabelsize=50)

sp = streamplot!(ax,B,-0.5..0.5,-0.5..0.5,stepsize=0.001,arrow_size=30)
cp = contour!(ax,XY,XY,Z,color=:black,linewidth=4.0)

hm = heatmap!(ax,XY,XY,Z,colormap=(:thermal,0.6))

Colorbar(p[1,2],hm,label=L"u(x,y,t_f)",labelsize=50)

axislegend(ax,[sp,cp],["B streamlines","u contours"])

save("NIMRODField.png", p)


