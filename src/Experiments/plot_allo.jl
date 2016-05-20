push!(LOAD_PATH, "../")
using JLD
using H2SDDP
using Gadfly
using DataFrames

function plot_allocation(γ,c)
  file = jldopen("./output/allo_data_G$(string(γ)[3:end])_C$(string(c)[3:end]).jld", "r")

  C = 3
  dH = read(file,"dH")
  dM = read(file,"dM")
  x = Array(Float64,dH.N+1,dH.T,C)
  u = Array(Float64,dH.N+1,dH.T,C)
  K = Array(Int64,dH.T,C)

  for s_f=1:C
    x[:,:,s_f] = read(file, "x$s_f")
    u[:,:,s_f] = read(file, "u$s_f")
    K[:,s_f] = read(file, "K$s_f")
  end
  close(file)

  x_p = Array(Float64,dH.T-1,C)
  y_p = Array(Float64,dH.T-1,C)
  sample = Array(Int64,dH.T-1,C)
  state = Array(Int64,dH.T-1,C)
  for i = 1:C
    for t = 2:dH.T
      x_p[t-1,i] = t
      y_p[t-1,i] = u[2,t,i]/sum(x[:,t,i])
      sample[t-1,i] = i
      state[t-1,i] = K[t,i]
    end
  end


  d = DataFrame(
    alloc  = reshape(y_p   , (dH.T-1)*C),
    time   = reshape(x_p   , (dH.T-1)*C),
    sample = reshape(sample, (dH.T-1)*C),
    state = reshape(state, (dH.T-1)*C)
  )

  # Plot
  p = plot(d,
    x=:time, y=:alloc, color=:state, Geom.point,
    Scale.discrete_color_manual("green","red","blue"))

  #draw(PDF("./output/figuras/allo_T$(dH.T)_C$(string(dH.γ)[3:end]).pdf", 4inch, 3inch),p )
  draw(PNG("./output/figuras/allo_T$(dH.T)_G$(string(dH.γ)[3:end])_C$(string(dH.c)[3:end]).png", 8inch, 6inch),p )
end
γs = [0.001; 0.01; 0.02]
for i = 1:size(γs,1)
  plot_allocation(γs[i],0.001)
end
