using MSDDP, HMM_MSDDP
using Distributions
using Logging

import FFM

Logging.configure(level=Logging.DEBUG)


function rollinghorizon(dH, series, nrows_train, F, R; real_tc=0.0, myopic=false)
  T_init = dH.T
  T_series = size(series,2)
  its=floor(Int,(T_series-nrows_train)/(R))

  all_x = vcat(dH.x0_ini,dH.x_ini)
  if R > dH.T-1
    error("R $R has to be last than or equal dH.T-1 $(dH.T-1)")
  end
  if myopic
    dH.T = 2
  end
  for i = 1:its
    ret_train = series[:,1:nrows_train+(i-1)*(R)]# TODO (i-1)*(R)+1:nrows_train+(i-1)*(R)
    ret_test  = series[:,nrows_train+(i-1)*(R):nrows_train+(i)*(R)]
    lnret_train = log(ret_train+1)
    lnret_test = log(ret_test+1)
    dFF = FFM.evaluate(lnret_train[1:F,end-R:end],lnret_train[F+1:end,end-R:end])
    dM, model = inithmm_ffm(lnret_train[1:F,:]', dFF, dH)

    LB, UB, LB_c, AQ, sp, x_trial, u_trial = sddp(dH, dM)

    info("Simulating")
    if myopic
      all = simulate_percport(dH, ret_test[F+1:end,:], u_trial[:,1]/sum(u_trial[:,1]))
      all_x = hcat(all_x,all[:,1:end])
    else
      dH.T = R+1
      states = predict(model,lnret_test[1:F,:]')
      x, x0, exp_ret = simulate(dH, dM, AQ, sp, ret_test[F+1:end,:], states; real_tc=real_tc)
      all_x = hcat(all_x,vcat(x0[2:end]',x[:,2:end]))
      dH.T = T_init
    end
    dH.x_ini = all_x[2:end,end]
    dH.x0_ini = all_x[1,end]
  end
  return all_x
end

function runMSDDP_TD_TC(dH, series, nrows_train, F, R)
  info("#### SDDP with temporal dependecy and transactional costs ####")
  all = rollinghorizon(dH, series, nrows_train, F, R)
  ret = sum(all,1)

  writecsv(string(output_dir,file_name,"_SDDP_TD_TC_k$(string(dH.K))g$(string(dH.γ)[3:end])_all.csv"),all)
  return ret
end

function runMSDDP_TD_NTC(dH, series, nrows_train, F, R)
  info("#### SDDP with temporal dependecy and no transactional costs ####")
  dH.c = 0
  @time all = rollinghorizon(dH, series, nrows_train, F, R; real_tc=c)
  ret = sum(all,1)

  writecsv(string(output_dir,file_name,"_SDDP_TD_NTC_k$(string(dH.K))g$(string(dH.γ)[3:end])_all.csv"),all)
  return ret
end

function runMyopic(dH, series, nrows_train, F, R)
  info("#### Myopic ####")
  @time all = rollinghorizon(dH, series, nrows_train, F, R; myopic=true)

  ret = sum(all,1)
  writecsv(string(output_dir,file_name,"_SDDP_My_k$(string(dH.K))g$(string(dH.γ)[3:end])_all.csv"),all)
  return ret
end

function runEqualy(dH, series, nrows_train, F)
  info("#### Equaly weight ####")

  info("Simulating")
  @time  all = simulate_percport(dH, series[F:end,nrows_train:end], ones(dH.N+1)*1/(dH.N+1))
  all = hcat(vcat(dH.x0_ini,dH.x_ini),all)
  ret = sum(all,1)

  writecsv(string(output_dir,file_name,"_SDDP_Eq_all.csv"),all)
  return ret
end

function runMSDDP_NTD_TC(dH, series, nrows_train, F, R)
  info("#### SDDP with no temporal dependecy and transactional costs ####")
  dH.K = 1
  @time all = rollinghorizon(dH, series, nrows_train, F, R)
  ret = sum(all,1)

  writecsv(string(output_dir,file_name,"_SDDP_NTD_TC_k$(string(dH.K))g$(string(dH.γ)[3:end])_all.csv"),all)
  return ret
end

srand(123)

#Parameters
N = 25
T = 12
K = 3
S = 1000
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.005
M = 9999999
γ = 0.08
S_LB = 300
S_FB = 100
GAPP = 1
Max_It = 100
α_lB = 0.9
dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

output_dir = "../../output/"

# Read series
F =3 # number of factors
file_dir = "../../input/"
file_name = "3FF_BM25_Large"
file = string(file_dir,file_name,".csv")
series = readcsv(file, Float64)'

R = 6 # Regression for FFM and number o test samples(avoid border effect, has to be lower than dH.T-1)
nrows_train = 882 # 120 (1990 to 2000)  204(1990 to 2006)
its=floor(Int,(size(series,2)-nrows_train)/(R))
series = series[:,1:nrows_train+(R)*its]
ret = zeros(Float64,5,(R)*its+1)

info("K = $(dH.K)")

ret[1,:] =       runEqualy(deepcopy(dH), series, nrows_train, F)
ret[2,:] =       runMyopic(deepcopy(dH), series, nrows_train, F, R)
ret[3,:] =  runMSDDP_TD_TC(deepcopy(dH), series, nrows_train, F, R)
ret[4,:] = runMSDDP_TD_NTC(deepcopy(dH), series, nrows_train, F, R)
ret[5,:] = runMSDDP_NTD_TC(deepcopy(dH), series, nrows_train, F, R)
writecsv(string(output_dir,file_name,"_ret_k$(string(dH.K))g$(string(dH.γ)[3:end]).csv"),ret)

run(`/home/tas/woofy.sh 62491240 "simulate"`)
