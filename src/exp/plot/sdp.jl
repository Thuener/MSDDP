using DataFrames, Gadfly
T = 12
table = readcsv("3MS_240_1000_$(T)_table.csv")
df2 = DataFrame()

df2[symbol("Trans. cost")] = vec(vcat(table[:,1],table[:,1],table[:,1]))
df2[:γ] = vec(vcat(table[:,2],table[:,2],table[:,2]))
df2[:OF] = vec(vcat(table[:,3],table[:,4],table[:,5]))

N = 12 # Numer of lines on table.csv
df2[:type] = vcat(fill("MSDDP",N),fill("One-Step",N),fill("One-Step M.",N))
df2[:M] = vec(vcat((table[:,3]+1).^(1/T) -1,(table[:,4]+1).^(1/T) -1,(table[:,5]+1).^(1/T) -1))

df2[:Mmin] = vec(vcat((table[:,6]+1).^(1/T) -1,(table[:,7]+1).^(1/T) -1,(table[:,8]+1).^(1/T) -1))
df2[:Mmax] = vec(vcat((table[:,9]+1).^(1/T) -1,(table[:,10]+1).^(1/T) -1,(table[:,11]+1).^(1/T) -1))

coord = Coord.cartesian(ymin=0.000, ymax=0.015)

myplot =  plot(df2,ygroup="type",x="γ",y="M", ymin="Mmin", ymax="Mmax",color="Trans. cost",Geom.subplot_grid(coord,Geom.line,Geom.errorbar)
          ,Scale.color_discrete(),Guide.ylabel("Monthly Expected Return"),Theme(line_width=2px,grid_line_width=2px))

draw(SVG("myplot3.svg", 9inch, 9inch), myplot) # paper
