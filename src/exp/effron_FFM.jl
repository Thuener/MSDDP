# Simulate AR model from Brown
using MSDDP, HMM_MSDDP, FFM, Util
import OneStep_FFM, SDP_FFM
using Distributions, Logging
Logging.configure(level=Logging.DEBUG)

function runMSDDP(dH, dM, Sc, rets_, states, ret_p)
  info("Memory use $(memuse())")
  ############ MSDDP ###########
  info("#### MSDDP ####")
  info("Training MSDDP...")
  @time LB, UB, LB_c, AQ, sp, = sddp(dH, dM;stabUB=0.05)

  info("Simulating MSDDP...")
  for se = 1:Sc
    x, x0 = MSDDP.forward(dH, dM, AQ, sp, states[:,se], rets_[:,:,se])
    ret_p[1,se] = x0[end]+sum(x[:,end])-1
  end
  info("Memory use $(memuse())")
  LB =0; UB =0; LB_c =0; AQ =0; sp =0;
  x  =0; x0  =0; exp_ret  =0;
end

function runOS(dH, dM, Sc, rets_, states, ret_p)
  ############ One Step ############
  info("#### One Step ####")
  info("Training One Step...")
  Q_t = ones(Float64, dH.T, dH.K)
  AQ, sp = OneStep_FFM.createmodels( dH, dM, Q_t )


  info("Simulating One Step...")
  for se = 1:Sc
    x, x0 = MSDDP.forward(dH, dM, AQ, sp, states[:,se], rets_[:,:,se])
    ret_p[2,se] = x0[end]+sum(x[:,end])-1.0
  end
  info("Memory use $(memuse())")
  H_l =0; sp =0; x =0; x0 =0;
end

function runOSFC(dH, dM, Sc, rets_, states, ret_p)
  ############ One Step with future cost ############
  # Run MSDDP with no costs
  info("#### One Step FC ####")
  info("Training One Step FC...")
  V_t = SDP_FFM.backward(dH, dM) # Get value function without transactional cost
  AQ, sp = OneStep_FFM.createmodels(dH, dM, V_t) # One step with tc

  info("Simulating One Step FC...")
  for se = 1:Sc
    x, x0 = MSDDP.forward(dH, dM, AQ, sp, states[:,se], rets_[:,:,se])
    ret_p[3,se] = x0[end]+sum(x[:,end])-1.0
  end
  info("Memory use $(memuse())")
  u_l =0; Q_l =0; H_l =0; sp =0; x =0; x0 =0;
end

#########################################################################################################################

srand(123)
γs = [0.05,0.1, 0.2,0.3]
cs = [0.005,0.01,0.02]
Ts = [12]#,24,48]

include("parametersBM100.jl")
T = dH.T = Ts[end]

output_dir = "../../output/outputFFM/"

# Read series
F =5 # number of factors
file_dir = "../../input/"
file_name = "5FF_BM100_Large"
file = string(file_dir,file_name,".csv")
data = readcsv(file, Float64)'

lnret = log(data+1)
dFF = FFM.evaluate(lnret[1:F,:],lnret[F+1:end,:])
dM, model = inithmm_ffm(lnret[1:F,:]', dFF, dH)

tic()


Sc = 1000
rets_f  = zeros(Float64,dH.N, dH.T)
states_f = Array(Int64, dH.T)
rets    = Array(Float64, dH.N, dH.T, Sc)
states  = Array(Int64, dH.T, Sc)

for se = 1:Sc
  simulatestates(dH, dM, states_f, rets_f)
  rets[:,:,se] = copy(rets_f)
  states[:,se] = copy(states_f)
end


for dH.T in Ts
  rets_ = rets[:,1:dH.T,:]
  states_ = states[1:dH.T,:]
  rets_p = zeros(3*3,length(γs),length(cs))
  file = string(output_dir,file_name,"_$(dH.T)_table.csv")
  for i_c = 1:length(cs)
    dH.c = cs[i_c]
    for i_γ = 1:length(γs)
      dH.γ = γs[i_γ]
      ret_p = Array(Float64,3,Sc)
      info("Start testes with γ = $(dH.γ), c = $(dH.c) and T = $(dH.T)")
      runMSDDP(dH, dM, Sc, rets_, states_, ret_p)
      gc()
      info("Memory use $(memuse())")
      runOS(dH, dM, Sc, rets_, states_, ret_p)
      gc()
      info("Memory use $(memuse())")
      runOSFC(dH, dM, Sc, rets_, states_, ret_p)
      gc()
      info("Memory use $(memuse())")
      for i = 1:3
        m = mean(ret_p[i,:])
        rets_p[i,i_γ,i_c] = m
        rets_p[i+3,i_γ,i_c] = m - (quantile(Normal(),0.975) * std(ret_p[i,:]) / sqrt(Sc))
        rets_p[i+6,i_γ,i_c] = m + (quantile(Normal(),0.975) * std(ret_p[i,:]) / sqrt(Sc))
      end
      open(file,"a") do x
        writecsv(x,hcat(dH.c, dH.γ,rets_p[:,i_γ,i_c]'))
      end
    end
  end
end
toc()

run(`/home/tas/woofy.sh 62491240 "simulateAR"`)
