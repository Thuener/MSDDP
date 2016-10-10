using DataFrames, Gadfly
#table = readcsv("/home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/outputIN2/5FF_BM25_Daily_90a15_ret.csv",Float64)
table = readcsv("/home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/outputIN/5FF_BM25_Large_ret.csv",Float64)
table = table[13:276,:]


M = 22 # number of tests
T = 11

data = Array(Float64,M*T,4)

for i = 1:M
	for t = 1:T
    ind_table = (i-1)*(T+1)+1
    ind_data = (i-1)*T+t
    data[ind_data,1] = table[ind_table+t,1]
		data[ind_data,2] = table[ind_table,1]
		data[ind_data,3] = table[ind_table,2]
		ret_t = table[ind_table+t,2]-1
		ret_T = table[ind_table+T,2]-1
    if ret_T-ret_t != 0
      data[ind_data,4] = ((ret_T-ret_t)/ret_T)*100
    else
      data[ind_data,4] = 0
    end
	end
end

df = DataFrame()
df[:T] = data[:,1]
df[:γ] = data[:,2]
tc = Array(String,size(data,1))
for i=1:size(data,1)
	tc[i] = string("c = ",data[i,3])
end
df[:c] = tc
df[:GAP] = data[:,4]

sort!(df, cols = [:γ])

#df = df[df[:c] .== 0.02,:]

coord = Coord.cartesian(xmin=Int(minimum(df[:T])), xmax=Int(maximum(df[:T])), ymin =minimum(df[:GAP]),ymax = maximum(df[:GAP]))

myplot =  plot(layer(df,ygroup="c",x="T",y="GAP", color="γ",
    Geom.subplot_grid(coord,Guide.yticks(ticks=collect(-40:20:100)),Geom.line)),
    Scale.color_discrete(),Guide.ylabel("GAP(%)"),Theme(line_width=2px,grid_line_width=2px,background_color=colorant"white"));

draw(PNG("bestT.png", 9inch, 9inch), myplot)

df2 = df[df[:T] .>= 4,:]
coord = Coord.cartesian(xmin=Int(minimum(df2[:T])), xmax=Int(maximum(df2[:T])), ymin =minimum(df2[:GAP]),ymax = maximum(df2[:GAP]))
myplot =  plot(layer(df2,ygroup="c",x="T",y="GAP", color="γ",
    Geom.subplot_grid(coord,Geom.line)),
    Scale.color_discrete(),Guide.ylabel("GAP(%)"),Theme(line_width=2px,grid_line_width=2px));

draw(PNG("bestT_zoom.png", 9inch, 9inch), myplot)
draw(PDF("bestT_zoom.pdf", 9inch, 9inch), myplot)
