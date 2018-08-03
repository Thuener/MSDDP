using DataFrames, Gadfly

println(ARGS)
file = ARGS[1] # Name of the file to be processed
single = parse(Bool,ARGS[2]) # if it is a single efficient frontier (true or false)
if length(ARGS) < 3
    table = readcsv("$(file)")[2:end,:]
else
    header = parse(Bool,ARGS[3]) # true or false to ignore header
    if header
        table = readcsv("$(file)")[2:end,:]
    else
        table = readcsv("$(file)")
    end
end
K = floor(Int,(size(table,2) -2)/3)
df2 = DataFrame()

N = size(table,1)  # Numer of lines on table.csv
names = ["MSDDP","FBS","FCBS"]

tc    = []
gamma = []
of    = []
ty    = []
M     = []
mmin  = []
mmax  = []
for i = 1:K
    tc    = vcat(tc,    table[:,1])
    gamma = vcat(gamma, table[:,2])
    of    = vcat(of,    table[:,i+2])

    ty = vcat(ty, fill(names[i],N))
    mmin = vcat(mmin, table[:,i+2+K])
    mmax = vcat(mmax, table[:,i+2+2*K])
end

df2[Symbol("Trans. cost")] = tc
df2[:γ]    = gamma
df2[:OF]   = of
df2[:type] = ty
df2[:Mmin] = mmin
df2[:Mmax] = mmax

myplot = []
if !single
  coord = Coord.cartesian(ymin=0.000, ymax=0.151)
  myplot =  plot(df2,ygroup="type",x="γ",y="OF", ymin="Mmin", ymax="Mmax",color="Trans. cost",Geom.subplot_grid(coord,Geom.line,Geom.errorbar)
          ,Scale.color_discrete_manual("lightgray","gray","black"),Guide.ylabel("Expected return (in 12 months)"),Theme(line_width=2px,grid_line_width=2px))
  draw(SVG("effron.svg", 9inch, 9inch), myplot) # paper
else
  myplot =  plot(df2,x="γ",y="OF", ymin="Mmin", ymax="Mmax",color="type",Guide.colorkey(""),Geom.line,Geom.errorbar
            ,Scale.color_discrete_manual("lightgray","gray","black"),Guide.ylabel("Expected return (in 12 months)"),Theme(line_width=2px,grid_line_width=2px))
  draw(SVG("singleeffron.svg", 9inch, 6inch), myplot) # paper
end
