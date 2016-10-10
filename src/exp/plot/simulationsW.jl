using Gadfly,Compose, DataFrames

################ PLot simulations W ################
function createDF(rets,γ)
  NS,days = size(rets)
  df = DataFrame()
  df[:x] = collect(1:days)
  df[:y] = squeeze(mean(rets,1),1)
  df[:f] = γ
  y = Array(Float64,days,3)
  for i = 1:days
    y[i,1] = quantile(rets[:,i],0.95)
    y[i,2] = quantile(rets[:,i],0.5)
    y[i,3] = quantile(rets[:,i],0.05)
  end
  df[:ymin] = y[:,3]
  df[:ymax] = y[:,1]
  df[:ymed] = y[:,2]
  return df
end
γ = 0.005
γ_srt = string(γ)[3:end]
rets1 = readcsv("5FF_BM25_Large_rets_g$(γ_srt)_c02.csv")
df1 = createDF(rets1,γ)

γ = 0.02
γ_srt = string(γ)[3:end]
rets2 = readcsv("5FF_BM25_Large_rets_g$(γ_srt)_c02.csv")
df2 = createDF(rets2,γ)

df = vcat(df2,df1)

dash = 0.3 * Compose.cm

l1=layer(df, x=:x,y=:ymed,color=:f,Geom.line,Theme(line_style=[dash]))
l2=layer(df, x=:x, y=:y, ymin=:ymin, ymax=:ymax, color=:f, Geom.line,Geom.ribbon,Theme(lowlight_opacity=0.5))

p = plot(l1,l2,Guide.xticks(ticks=collect(2:1:5)),Scale.color_discrete(),Theme(grid_line_width=2px))

draw(PDF("SW.png", 9inch, 9inch), p)

## ploting without ribbon
l1=layer(df1, x=:x,y=:ymed,color=:f,Geom.line,Theme(line_style=[dash]))
l2=layer(df1, x=:x,y=:ymax,color=:f,Geom.line,Theme())
l3=layer(df1, x=:x,y=:ymin,color=:f,Geom.line,Theme())

l4=layer(df2, x=:x,y=:ymed,color=:f,Geom.line,Theme(line_style=[dash]))
l5=layer(df2, x=:x,y=:ymax,color=:f,Geom.line,Theme())
l6=layer(df2, x=:x,y=:ymin,color=:f,Geom.line,Theme())

p = plot(l1,l2,l3,l4,l5,l6,Guide.xticks(ticks=collect(2:1:5)),Scale.color_discrete(),Theme(grid_line_width=2px))
draw(PNG("SW.png", 9inch, 9inch), p)
