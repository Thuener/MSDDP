# Plot the first stage decisions for S different instances of MSDDP
using DataFrames, Vega

println(ARGS)
file = ARGS[1] # Name of the file to be processed

list_firststage = readcsv(file)
names = [" RF","SMALL LoOP","ME1 OP2","ME1 OP3","ME1 OP4","ME1 OP5","ME1 OP6","ME1 OP7",
  "ME1 OP8","ME1 OP9","SMALL HiOP","ME2 OP1","ME2 OP2","ME2 OP3","ME2 OP4",
  "ME2 OP5","ME2 OP6","ME2 OP7","ME2 OP8","ME2 OP9","ME2 OP10","ME3 OP1",
  "ME3 OP2","ME3 OP3","ME3 OP4","ME3 OP5","ME3 OP6","ME3 OP7","ME3 OP8",
  "ME3 OP9","ME3 OP10","ME4 OP1","ME4 OP2","ME4 OP3","ME4 OP4","ME4 OP5",
  "ME4 OP6","ME4 OP7","ME4 OP8","ME4 OP9","ME4 OP10","ME5 OP1","ME5 OP2",
  "ME5 OP3","ME5 OP4","ME5 OP5","ME5 OP6","ME5 OP7","ME5 OP8","ME5 OP9",
  "ME5 OP10","ME6 OP1","ME6 OP2","ME6 OP3","ME6 OP4","ME6 OP5","ME6 OP6",
  "ME6 OP7","ME6 OP8","ME6 OP9","ME6 OP10","ME7 OP1","ME7 OP2","ME7 OP3",
  "ME7 OP4","ME7 OP5","ME7 OP6","ME7 OP7","ME7 OP8","ME7 OP9","ME7 OP10",
  "ME8 OP1","ME8 OP2","ME8 OP3","ME8 OP4","ME8 OP5","ME8 OP6","ME8 OP7",
  "ME8 OP8","ME8 OP9","ME8 OP10","ME9 OP1","ME9 OP2","ME9 OP3","ME9 OP4",
  "ME9 OP5","ME9 OP6","ME9 OP7","ME9 OP8","ME9 OP9","ME9 OP10","BIG LoOP",
  "ME10 OP2","ME10 OP3","ME10 OP4","ME10 OP5","ME10 OP6","ME10 OP7",
  "ME10 OP8","ME10 OP9","BIG HiOP"]

# get only the assets that have some allocation
names = names[vec(sum(list_firststage,2) .!= 0)]
list_firststage = list_firststage[vec(sum(list_firststage,2) .!= 0),:] # get only the assets that have some allocation
df = DataFrame()
N = size(list_firststage,1)
S = size(list_firststage,2)
df[:it] = repeat(collect(1:S),inner=[N])
df[:as] = repeat(names,outer=[S])
df[:al] = vec(list_firststage)


a = barplot(x = df[:it], y = df[:al], group = df[:as], stacked = true, normalize = true)
legend!(a, title = "Portfolios",show=true)
xlab!(a, title = "MSDDP")
ylab!(a, title = "Percentage allocation")


#=
Geom.boxplot
plot(df, y=:al, x=:it, color=:as,Geom.bar,
  Guide.ylabel("Percentage allocation"),Scale.color_discrete(),Scale.discrete_color_manual("blue","red", "green"),
  Guide.xticks(ticks=collect(1:S)))
=#
