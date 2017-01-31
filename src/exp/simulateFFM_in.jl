# Simulate the Farma French Model in sample

using MSDDP, HMM_MSDDP, Util
using Distributions
using Logging

import FFM

Logging.configure(level=Logging.DEBUG)

function rollinghorizon(dH, dM, states, series, R; real_tc=0.0, myopic=false)
  T_init = dH.T
  T_series = size(series,2)
  its=floor(Int,(T_series)/(R))

  all_x = vcat(dH.x0,dH.x)
  if R > dH.T-1
    error("R $R has to be last than or equal dH.T-1 $(dH.T-1)")
  end
  for i = 1:its
    ret_it = series[:,1:(i-1)*(R)]
    states_it = states[1:(i-1)*(R)]

    info("Simulating $i of $its memuse $(memuse())")
    if myopic
      all = simulate_percport(dH, ret_it[:,:], u_trial[:,1]/sum(u_trial[:,1]))
      all_x = hcat(all_x,all[:,2:end])
    else
      dH.T = R+1
      debug("States $(states)")
      x, x0, exp_ret = simulate(dH, dM, AQ, sp, ret_it[:,:], states; real_tc=real_tc)
      all_x = hcat(all_x,vcat(x0[2:end]',x[:,2:end]))
      dH.T = T_init
    end
    dH.x = all_x[2:end,end]
    dH.x0 = all_x[1,end]
  end
  return all_x
end

function runMSDDP_TD_TC(dH, dM, series, states)
  info("#### SDDP with temporal dependecy and transactional costs ####")
  NS = size(series,3)

  LB, UB, LB_c, AQ, sp, x_trial, u_trial = sddp(dH, dM)
  rets = Array(Float64, NS, dH.T-1)
  for s = 1:NS
    x, x0, exp_ret = simulate(dH, dM, AQ, sp, series[:,:,s], states[:,s])
    all = vcat(x0[2:end]',x[:,2:end])
    rets[s,:] = sum(all,1)
  end
  return mean(rets)
end

function runMSDDP_TD_NTC(dH, dM, series, states)
  info("#### SDDP with temporal dependecy and no transactional costs ####")
  NS = size(series,3)

  c = dH.c
  dH.c = 0.0

  LB, UB, LB_c, AQ, sp, x_trial, u_trial = sddp(dH, dM)

  rets = Array(Float64, NS, dH.T-1)
  for s = 1:NS
    x, x0, exp_ret = simulate(dH, dM, AQ, sp, series[:,:,s], states[:,s]; real_tc=c)
    all = vcat(x0[2:end]',x[:,2:end])
    rets[s,:] = sum(all,1)
  end
  return mean(rets)
end

function runMyopic_Inc_TC(dH, series, ret_inc)
  info("#### Myopic inconditional TC ####")
  NS = size(series,3)

  T_ini = dH.T
  dH.T = 2
  dH.K = 1
  dH.S = NS

  p_s = ones(dH.S, dH.K)*1.0/dH.S
  dM = MKData( ret_inc, p_s, 1, [1.0]' )
  LB, UB, LB_c, AQ, sp, x_trial, u_trial = sddp(dH, dM)

  rets = Array(Float64, NS, T_ini-1)
  for s = 1:NS
    all = simulate_percport(dH, series[:,:,s], u_trial[:,1]/sum(u_trial[:,1]))
    rets[s,:] = sum(all[:,2:end],1)
  end
  return mean(rets)
end

function runMSDDP_NTD_TC(dH, series, ret_inc)
  info("#### SDDP with no temporal dependecy and transactional costs ####")
  NS = size(series,3)

  dH.K = 1
  dH.S = NS

  p_s = ones(dH.S, dH.K)*1.0/dH.S
  dM = MKData( ret_inc, p_s, 1, [1.0]')
  LB, UB, LB_c, AQ, sp, x_trial, u_trial = sddp(dH, dM)

  rets = Array(Float64, NS, dH.T-1)
  states = ones(Int64,dH.T)
  for s = 1:NS
    x, x0, exp_ret = simulate(dH, dM, AQ, sp, series[:,:,s], states)
    all = vcat(x0[2:end]',x[:,2:end])
    rets[s,:] = sum(all,1)
  end
  return mean(rets)
end


function runMyopic_NTC(dH, dM, series, states)
  info("#### Myopic NTC ####")
  NS = size(series,3)

  T_ini = dH.T
  dH.T = 2
  c = dH.c
  dH.c = 0.0

  LB, UB, LB_c, AQ, sp, x_trial, u_trial = sddp(dH, dM)

  rets = Array(Float64, NS, T_ini-1)
  for s = 1:NS
    x, x0 = simulatesw(dH, dM, AQ, sp, series[:,:,s], states[:,s]; real_tc=c)
    all = vcat(x0[2:end]',x[:,2:end])
    rets[s,:] = sum(all,1)
  end
  return mean(rets)
end

srand(123)

#Parameters
N = 25
T = 5 #TODO 22
K = 3 #TODO 5
S = 500 #TODO 1000
α = 0.9
W_ini = 1.0
x_ini_s = [W_ini;zeros(N)]
c = 0.005
M = 9999999
γ = 0.02
S_LB = 300
S_LB_inc = 100
S_FB = 10 #TODO 20
GAPP = 1
Max_It = 1#TODO 100
α_lB = 0.9
dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], W_ini, c, M, γ,
                S_LB, S_LB_inc, S_FB, GAPP, Max_It, α_lB )

output_dir = "../../outputIN/"

# Read series
F =5 # number of factors
file_dir = "../../input/"
file_name = "5FF_BM25_Daily_90a15"
file = string(file_dir,file_name,".csv")
data = readcsv(file, Float64)'

#R = 21 # Regression for FFM and number o test samples(avoid border effect, has to be lower than dH.T-1)
lnret = log(data+1)
dFF = FFM.evaluate(lnret[1:F,:],lnret[F+1:end,:])
dM, model = inithmm_ffm(lnret[1:F,:]', dFF, dH)

NS = 1000
series = zeros(Float64,N,T,NS)
states = zeros(Int64,T,NS)
K_forward = Array(Int64,dH.T)
r_forward = zeros(dH.N,dH.T)
for s = 1:NS
  simulatestates(dH, dM, K_forward, r_forward)
  series[:,:,s] = r_forward
  states[:,s] = K_forward
end
ret_inc = series[:,end-2,:]

info("dH = $dH")

ret = zeros(Float64,5)

ret[1]= runMyopic_Inc_TC(deepcopy(dH), series, ret_inc)
ret[2] = runMSDDP_NTD_TC(deepcopy(dH), series, ret_inc)

ret[3] = runMyopic_NTC(deepcopy(dH), dM, series, states)
ret[4]=runMSDDP_TD_NTC(deepcopy(dH), dM, series, states)

ret[5]= runMSDDP_TD_TC(deepcopy(dH), dM, series, states)

writecsv(string(output_dir,file_name,"_ret.csv"),vcat(dH.γ,dH.c,ret))

run(`/home/tas/woofy.sh 62491240 "Finish $(@__FILE__) "`)
