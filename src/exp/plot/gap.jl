using Gadfly,Compose, DataFrames
################ PLot GAP ################

LB = readcsv("5FF_BM100_Large_LB_g06_c02.csv")
UB = readcsv("5FF_BM100_Large_UB_g06_c02.csv")

df = DataFrame()

df[:LB] = LB[2:end,1]
df[:uLB] = LB[2:end,1]+ 	1.96* LB[2:end,2]
df[:lLB] = LB[2:end,1]- 	1.96* LB[2:end,2]

df[:UB] = UB[2:end]
df[:It] = collect(1:size(UB,1)-1)

dash = 0.3 * Compose.cm
l1=layer(df, x=:It,y=:UB,Geom.line, Theme(default_color=colorant"red",line_style=[dash]))
l2=layer(df, x=:It, y=:LB, ymin=:lLB, ymax=:uLB, Geom.line, Geom.ribbon)

coord = Coord.cartesian(xmin=1, xmax=size(UB,1)-1)
p = plot(l1,l2, coord,
  Guide.ylabel("Objective value"),Guide.xticks(ticks=collect(1:1:maximum(df[:It]))),Guide.xlabel("MSDDP iteration"),
  Theme(line_width=2px,grid_line_width=1px),
  Guide.manual_color_key("",["UB","LB"], ["red", "deepskyblue"]));

draw(SVG("gap_convergence.svg", 16cm, 10cm), p)
