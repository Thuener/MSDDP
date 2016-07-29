using MSDDP
using Distributions
using Base.Test
using Logging
using Gadfly

N = 1
T = 10
K = 3
S = 100
α = 0.9
x_ini = zeros(N)
x0_ini = 1.0
c = 0.00
M = 9999999#(1+maximum(r))^(T-1)*(sum(x_ini)+x0_ini)-(sum(x_ini)+x0_ini);
γ = 1
#Parameters
S_LB = 500
S_FB = 5
GAPP = 1 # GAP mínimo em porcentagem
Max_It = 100
α_lB = 0.9

dH  = MSDDPData( N, T, K, S, α, x_ini, x0_ini, c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

file = "../C++/output/ken_1DInd_90a14"

dM = readHMMPara(file, dH)

pdfs = Vector{Function}(dH.K)
dists = Vector{Normal}(dH.K)
x = :x
for j = 1:dH.K
  ret = reshape(dM.r[1,j,:],100)
  dists[j] = fit(Normal, ret)
  pdfs[j] =@eval $x->pdf(dists[$j],$x)
end
color=[string(j) for j = 1:dH.K]
allret = reshape(dM.r,300)
p = plot(pdfs,minimum(allret),maximum(allret),color=color,Guide.colorkey("State"))
draw(PDF("./output/figuras/normalmixture.pdf", 4inch, 3inch),p )
