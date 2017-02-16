using DataFrames, Gadfly

println(ARGS)
file = ARGS[1] # Name of the file to be processed
single = parse(Bool,ARGS[2]) # if it is a single efficient frontier (true or false)

table = readcsv("$(file)")
df2 = DataFrame()

df2[Symbol("Trans. cost")] = vec(vcat(table[:,1],table[:,1],table[:,1]))
df2[:γ] = vec(vcat(table[:,2],table[:,2],table[:,2]))
df2[:OF] = vec(vcat(table[:,3],table[:,4],table[:,5]))

N = size(table,1) # Numer of lines on table.csv
df2[:type] = vcat(fill("MSDDP",N),fill("One-Step",N),fill("One-Step M.",N))
df2[:M] = vec(vcat((table[:,3]),(table[:,4]),(table[:,5])))

df2[:Mmin] = vec(vcat((table[:,6]),(table[:,7]),(table[:,8])))
df2[:Mmax] = vec(vcat((table[:,9]),(table[:,10]),(table[:,11])))

myplot =0
if !single
  coord = Coord.cartesian(ymin=0.000, ymax=0.151)
  myplot =  plot(df2,ygroup="type",x="γ",y="M", ymin="Mmin", ymax="Mmax",color="Trans. cost",Geom.subplot_grid(coord,Geom.line,Geom.errorbar)
          ,Scale.color_discrete(),Guide.ylabel("Expected return (in 12 months)"),Theme(line_width=2px,grid_line_width=2px))
  draw(SVG("effron.svg", 9inch, 9inch), myplot) # paper
else
  myplot =  plot(df2,x="γ",y="M", ymin="Mmin", ymax="Mmax",color="type",Guide.colorkey(""),Geom.line,Geom.errorbar
            ,Scale.color_discrete(),Guide.ylabel("Expected return (in 12 months)"),Theme(line_width=2px,grid_line_width=2px))
  draw(SVG("singleeffron.svg", 9inch, 6inch), myplot) # paper
end
