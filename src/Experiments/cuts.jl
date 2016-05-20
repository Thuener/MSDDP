push!(LOAD_PATH, "../")
using H2SDDP
using Distributions
using Base.Test
using Logging
using Gadfly
using HDF5
using JLD

#=
function piecewise(x::Symbol,B::Array{Float64,2},A::Array{Float64,1})
  @assert size(B,2) == 2
  @assert size(A,1) == size(B,1)
  vf= @eval $x->(1-$x)*$B[:,1] + $x*$B[:,2] + $A
  return @eval $x->minimum($vf($x))
end
=#
srand(123)

function piecewise(w::Float64,B::Array{Float64,2},A::Array{Float64,1})
  println("B = $B")
  println("A = $A")
  @assert size(B,2) == 2
  @assert size(A,1) == size(B,1)
  f(x_1,x_2) = minimum(x_1*B[:,1] + x_2*B[:,2] + A)
  f2(x_2) = f((1-x_2),x_2)
  return f2
end

function evafunc(dH::H2SDDPData, dM::HMMData, α, β, t, k )
  n_cuts = size(α,1)
  a = Array(Float64,n_cuts)
  b = Array(Float64,dH.N+1,n_cuts)
  pf = Array{Function,1}(dH.K)
  for j = 1:dH.K
    for i = 1:n_cuts
      a[i] = α[i][t,j]
      b[:,i] = β[i][:,t,j]
    end
    pf[j]=piecewise(1.0,b', a)
#    p = plot(pf[j],0.0,1)
#    draw(PNG("./output/figuras/cuts_T$(T)_k$(k)_j$(j).png", 4inch, 3inch),p )
  end
  return pf
end

N = 1
T = 10
K = 3
S = 100
α = 0.9
x_ini = zeros(N)
x0_ini = 1.0
c = 0.001
M = 9999999#(1+maximum(r))^(T-1)*(sum(x_ini)+x0_ini)-(sum(x_ini)+x0_ini);
γ = 1
#Parâmetros
S_LB = 500
S_FB = 5
GAPP = 1 # GAP mínimo em porcentagem
Max_It = 100
α_lB = 0.9

dH  = H2SDDPData( N, T, K, S, α, x_ini, x0_ini, c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

file = "../C++/output/ken_1DInd_90a14"

Logging.configure(level=INFO, filename="./log/cuts$(T).log")

γs = [0.001; 0.01; 0.02]
#γs = [0.0009; 0.0005; 0.0001]
tic()
#readall(`../C++/HMM /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $K $N $S 1`)
pf = Array(Function,dH.K,size(γs,1))
for k = 1:size(γs,1)
    println("************************$(γs[k])************************")
    dM = readHMMPara(file, dH)
    dH.γ = γs[k]
    dH.S_LB = S_LB
    LB, UB, AQ, sp, list_α, list_β, x_trial, u_trial = SDDP(dH, dM, simuLB=true)
    jldopen("./output/cuts_$(k)_G$(string(dH.γ)[3:end])_C$(string(dH.c)[3:end]).jld", "w") do file
      write(file, "u", u_trial)
      write(file, "x", x_trial)
      write(file, "beta", list_β)
      write(file, "alpha", list_α)
    end
    pf[:,k] = evafunc(dH, dM, list_α, list_β, 2, k)
end

color=[string(γs[i]) for i = 1:size(γs,1)]
for j = 1:dH.K
  p = plot(reshape(pf[j,:],3),0.0,1,color=color,Guide.colorkey("γ"))
  draw(PDF("./output/figuras/cuts_T$(T)_j$(j)_C$(string(c)[3:end]).pdf", 4inch, 3inch),p )
end

toc()
