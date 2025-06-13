module Simulate

using MSDDP, HMM_MSDDP
using Distributions
using Logging

import FFM
export runMSDDP_TD_TC, runMSDDP_TD_NTC, runMyopic, runEqualy, runMSDDP_NTD_TC
export runMSDDP_TD_TC_IN

Logging.configure(level=Logging.DEBUG)


function rollinghorizon(dH, series, nrows_train, F, R, output_dir, file_name; real_tc=0.0, myopic=false)
  T_init = dH.T
  T_series = size(series,2)
  its=floor(Int,(T_series-nrows_train)/(R))

  all_x = vcat(dH.x0,dH.x)
  if R > dH.T-1
    error("R $R has to be last than or equal dH.T-1 $(dH.T-1)")
  end
  if myopic
    dH.T = 2
  end
  for i = 1:its
    ret_train = series[:,1:nrows_train+(i-1)*(R)]# (i-1)*(R)+1:nrows_train+(i-1)*(R)
    ret_test  = series[:,nrows_train+(i-1)*(R):nrows_train+(i)*(R)]
    lnret_train = log(ret_train+1)
    lnret_test = log(ret_test+1)
    dFF = FFM.evaluate(lnret_train[1:F,:],lnret_train[F+1:end,:])
    dM, model = inithmm_ffm(lnret_train[1:F,:]', dFF, dH)

    LB, UB, LB_c, x_trial, u_trial = sddp(dH, dM;stabUB=0.05)

    info("Simulating $i of $its memuse $(memuse())")
    states = predict(model,lnret_test[1:F,:]')
    if myopic
      x, x0 = simulatesw(dH, dM, AQ, sp, ret_test[F+1:end,:], states; real_tc=real_tc)
      all_x = hcat(all_x,vcat(x0[2:end]',x[:,2:end]))
    else
      dH.T = R+1
      debug("States $(states)")
      x, x0, exp_ret = simulate(dH, dM, AQ, sp, ret_test[F+1:end,:], states; real_tc=real_tc)
      all_x = hcat(all_x,vcat(x0[2:end]',x[:,2:end]))
      dH.T = T_init
    end
    dH.x = all_x[2:end,end]
    dH.x0 = all_x[1,end]
    writecsv(string(output_dir,file_name,"R$(R)k$(string(dH.K))g$(string(dH.γ)[3:end])_all.csv"),all_x')
  end
  return all_x
end

function runMSDDP_TD_TC(dH, series, nrows_train, F, R, output_dir, file_name)
  info("#### SDDP with temporal dependecy and transactional costs ####")
  all = rollinghorizon(dH, series, nrows_train, F, R, output_dir, string(file_name,"_SDDP_TD_TC_"))
  ret = sum(all,1)

  writecsv(string(output_dir,file_name,"_SDDP_TD_TC_R$(R)k$(string(dH.K))g$(string(dH.γ)[3:end])_all.csv"),all')
  return ret
end

function runMSDDP_TD_NTC(dH, series, nrows_train, F, R, output_dir, file_name)
  info("#### SDDP with temporal dependecy and no transactional costs ####")
  c = dH.c
  dH.c = 0
  @time all = rollinghorizon(dH, series, nrows_train, F, R, output_dir, string(file_name,"_SDDP_TD_NTC_"); real_tc=c)
  ret = sum(all,1)

  writecsv(string(output_dir,file_name,"_SDDP_TD_NTC_R$(R)k$(string(dH.K))g$(string(dH.γ)[3:end])_all.csv"),all')
  return ret
end

function runMyopic(dH, series, nrows_train, F, R, output_dir, file_name)
  info("#### Myopic ####")
  @time all = rollinghorizon(dH, series, nrows_train, F, R, output_dir, string(file_name,"_SDDP_My_"); myopic=true)

  ret = sum(all,1)
  writecsv(string(output_dir,file_name,"_SDDP_My_R$(R)k$(string(dH.K))g$(string(dH.γ)[3:end])_all.csv"),all')
  return ret
end

function runEqualy(dH, series, nrows_train, F, output_dir, file_name)
  info("#### Equaly weight ####")

  info("Simulating")
  @time  all = simulate_percport(dH, series[F+1:end,nrows_train:end], ones(dH.N+1)*1/(dH.N+1))
  ret = sum(all,1)

  writecsv(string(output_dir,file_name,"_SDDP_Eq_all.csv"),all)
  return ret
end

function runMSDDP_NTD_TC(dH, series, nrows_train, F, R, output_dir, file_name)
  info("#### SDDP with no temporal dependecy and transactional costs ####")
  dH.K = 1
  @time all = rollinghorizon(dH, series, nrows_train, F, R, output_dir, string(file_name,"_SDDP_NTD_TC_"))
  ret = sum(all,1)

  writecsv(string(output_dir,file_name,"_SDDP_NTD_TC_R$(R)k$(string(dH.K))g$(string(dH.γ)[3:end])_all.csv"),all')
  return ret
end

### Start part in sample
function runMSDDP_TD_TC_IN(dH, dM, series, states, max_T)
  info("#### SDDP with temporal dependecy and transactional costs T$(dH.T) ####")
  debug(dH)
  NS = size(series,3)
  days = size(series,2)

  LB, UB, LB_c, x_trial, u_trial = sddp(dH, dM)
  rets = Array(Float64, NS, days-1)
  for s = 1:NS
    x, x0 = simulatesw(dH, dM, AQ, sp, series[:,:,s], states[:,s])
    all = vcat(x0[2:end]',x[:,2:end])
    if size(x,2) != max_T && size(x0) != max_T
      erro("Worng number of days in simulatesw.")
    end
    rets[s,:] = sum(all,1)
  end
  ret = mean(rets[:,end])
  debug("ret $(ret)")
  return ret,rets
end

end
