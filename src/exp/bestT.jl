addprocs(5)
using MSDDP, HMM_MSDDP, Simulate, Util
using Distributions
using Logging

import FFM

Logging.configure(level=Logging.DEBUG)

srand(123)

#Parameters
N = 25
T = 0
K = 3
S = 1000
α = 0.9
W_ini = 1.0
x_ini_s = [W_ini;zeros(N)]
c = 0.02
M = 9999999
γ = 0.02
S_LB = 300
S_LB_inc = 100
S_FB = 10
GAPP = 1
Max_It = 100
α_lB = 0.9
dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], W_ini, c, M, γ,
                S_LB, S_LB_inc, S_FB, GAPP, Max_It, α_lB )

output_dir = "../../outputIN/"

# Read series
F =5 # number of factors
file_dir = "../../input/"
file_name = "5FF_BM25_Large"
file = string(file_dir,file_name,".csv")
data = readcsv(file, Float64)'

lnret = log(data+1)
dFF = FFM.evaluate(lnret[1:F,:],lnret[F+1:end,:])
dM, model = inithmm_ffm(lnret[1:F,:]', dFF, dH)

max_T = 12
min_T = 2
dH.T = max_T
NS = 1000
series = zeros(Float64,dH.N,dH.T,NS)
states = zeros(Int64,dH.T,NS)
K_forward = Array(Int64,dH.T)
r_forward = zeros(dH.N,dH.T)
for s = 1:NS
  simulatestates(dH, dM, K_forward, r_forward)
  series[:,:,s] = r_forward
  states[:,s] = K_forward
end
ret_inc = series[:,end-2,:]

info("dH = $dH")

γs = [0.005, 0.01,0.02]
cs = [0.005,0.01,0.02] #

file = string(output_dir,file_name,"_ret.csv")
# For each risk level (γ)
for i_γ = 1:length(γs)
  dH.γ = γs[i_γ]
  # For each transactional cost (c)
  for i_c = 1:length(cs)
    dH.c = cs[i_c]
    ret = SharedArray(Float64, max_T -min_T +1)
    @sync @parallel for T=min_T:max_T
      dH.T = T
      ret[T-min_T+1],rets = runMSDDP_TD_TC_IN(deepcopy(dH), dM, series, states, max_T)
    end
    open(file,"a") do x
      writecsv(x,vcat(hcat(dH.γ,dH.c),hcat(collect(min_T:max_T),ret)))
    end
  end
end

run(`/home/tas/woofy.sh 62491240 "bestT"`)
