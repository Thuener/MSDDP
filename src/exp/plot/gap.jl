using Gadfly, Compose, DataFrames, Distributions, JLD
################ PLot GAP ################
println(ARGS)
file = ARGS[1] # Name of the file to be processed

data = load(file)

x0_ini = 1000000.0
LB = data["l_LB"][1:end,:]/x0_ini
UB = data["l_UB"][1:end,:]/x0_ini

df = DataFrame()
q = quantile(Normal(),0.99)
df[:LB] = LB[2:end,1]
df[:uLB] = LB[2:end,1]+ 	q* LB[2:end,2]
df[:lLB] = LB[2:end,1]- 	q* LB[2:end,2]

df[:UB] = UB[2:end]
df[:It] = collect(1:size(UB,1)-1)*5

dash = 0.3 * Compose.cm
l1=layer(df, x=:It,y=:UB,Geom.line, Theme(default_color=colorant"red",line_style=[dash]))
l2=layer(df, x=:It, y=:LB, ymin=:lLB, ymax=:uLB, Geom.line, Geom.ribbon)

#coord = Coord.cartesian(xmin=1, xmax=size(UB,1)-1)
p = plot(l1,l2, #coord,
  Guide.ylabel("Objective value"),Guide.xticks(ticks=collect(5:5:maximum(df[:It]))),Guide.xlabel("MSDDP iteration(added cuts)"),
  Theme(line_width=2px,grid_line_width=1px),
  Guide.manual_color_key("",["UB","LB"], ["red", "deepskyblue"]))

draw(SVG("gap_convergence.svg", 16cm, 10cm), p)
