# Simulate AR model from Brown
using MSDDP, HMM_MSDDP, AR, Util
import OneStep, SDP
using Distributions, Logging
Logging.configure(level=Logging.DEBUG)

srand(123)
include("parametersAR.jl")
# AR
dF = ARData(a_z, a_r, b_z, b_r, Σ, r_f)

# Parmeters
dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

# One Step data
dO = OneStep.OSData(N, T, L, S, α, c, γ, true, x_ini_s[2:N+1], x_ini_s[1])
dS = SDP.SDPData(N, T, L, S, α, γ)

#########################################################################################################################
tic()
γs = [0.05,0.1, 0.2,0.3]
cs = [0.005,0.013,0.02]
Ts = [12]#,24,48]

# Split z
z_l = SDP.splitequaly(dS.L, z)

# HMM data
info("Train HMM")
z_slothmm = SDP.splitequaly(dH.K, z)
v_hmm = dF.Σ[dH.N+1,dH.N+1]*ones(dH.K)
dM, model = inithmm_ar(z[T_max-T_hmm+1:T_max,:], dF, dH, T_hmm, Sc, z_slothmm, v_hmm)

states = Array(Int64,Ts[end],Sc)
for se = 1:Sc
  states[:,se] = predict(model,z[1:Ts[end],se])
end

function runMSDDP(dH, dM, Sc, rets_, states, ret_p)
  info("Memory use $(memuse())")
  ############ MSDDP ###########
  info("#### MSDDP ####")
  info("Training MSDDP...")
  @time LB, UB, LB_c, AQ, sp, = sddp(dH, dM)

  info("Simulating MSDDP...")
  for se = 1:Sc
    x, x0, exp_ret = simulate(dH, dM, AQ, sp, rets_[:,:,se], states[:,se])
    ret_p[1,se] = x0[end]+sum(x[:,end])-1
  end
  info("Memory use $(memuse())")
  LB =0; UB =0; LB_c =0; AQ =0; sp =0;
  x  =0; x0  =0; exp_ret  =0;
end

function runOS(dF, dO, z_l, rets_, z, ret_p)
  ############ One Step ############
  info("#### One Step ####")
  info("Training One Step...")
  β = ones(Float64, dS.T, 3)

  info("Simulating One Step...")
  for se = 1:Sc
    x, x0 = OneStep.forward(dO, dF, dS, β, z_l, z[:,se], rets_[:,:,se])
    ret_p[2,se] = x0[end]+sum(x[:,end])-1.0
  end
  info("Memory use $(memuse())")
  H_l =0; sp =0; x =0; x0 =0;
end

function runOSFC(dF, dS, z_l, rets_, z, ret_p)
  ############ One Step with future cost ############
  # Run MSDDP with no costs
  info("#### One Step FC ####")
  info("Training One Step FC...")
  dO.Mod = true
  # Run SDP
  @time β = SDP.backward(dF, dS, z_l)

  info("Simulating One Step FC...")
  for se = 1:Sc
    x, x0 = OneStep.forward(dO, dF, dS, β, z_l, z[:,se], rets_[:,:,se])
    ret_p[3,se] = x0[end]+sum(x[:,end])-1.0
  end
  info("Memory use $(memuse())")
  u_l =0; Q_l =0; H_l =0; sp =0; x =0; x0 =0;
end

for dH.T in Ts
  dS.T = dO.T = dH.T
  rets_ = rets[:,1:dH.T,:]
  rets_p = zeros(3*3,length(γs),length(cs))
  file = string(output_dir,file_name,"_$(dH.T)_table.csv")
  for i_c = 1:length(cs)
    dH.c = dO.c = cs[i_c]
    for i_γ = 1:length(γs)
      dH.γ = dS.γ = dO.γ = γs[i_γ]
      ret_p = SharedArray(Float64,3,Sc)
      info("Start testes with γ = $(dH.γ), c = $(dH.c) and T = $(dH.T)")
      runMSDDP(dH, dM, Sc, rets_, states, ret_p)
      gc()
      info("Memory use $(memuse())")
      dO.Mod = false
      runOS(dF, dO, z_l, rets_, z, ret_p)
      gc()
      info("Memory use $(memuse())")
      runOSFC(dF, dS, z_l, rets_, z, ret_p)
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
