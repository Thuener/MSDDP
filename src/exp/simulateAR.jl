using MSDDP, HMM_MSDDP, AR
import OneStep, SDP
using Distributions, Logging
Logging.configure(level=Logging.DEBUG)

srand(123)
T_max = 240
Sc = 1000
T_hmm = 120

# AR
Σ = [0.002894	0.003532	0.00391	-0.000115; 0.003532	0.004886	0.005712	-0.000144; 0.00391	0.005712	0.007259	-0.000163; -0.000115	-0.000144	-0.000163	0.0529]
b_r = [ 0.0028; 0.0049; 0.0061]
b_z = [0.9700]
a_r  = [0.0053; 0.0067; 0.0072]
a_z  = [0.0000]
r_f = 0.00042
dF = ARData(a_z, a_r, b_z, b_r, Σ, r_f)

# Parmeters
N = 3
T = 13
K = 4
S = 750
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.005
M = 9999999
γ = 0.002
S_LB = 300
S_FB = 100
GAPP = 1
Max_It = 15
α_lB = 0.9

dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )

# One Step data
L = 1000
dO = OneStep.OSData(N, T, L, S, α, c, γ, true, x_ini_s[2:N+1], x_ini_s[1])
dS = SDP.SDPData(N, T, L, S, α, γ)

#########################################################################################################################
tic()
γs = [0.05,0.1, 0.2,0.3] #0.05,0.1, 0.2,0.3
cs = [0.005,0.01,0.02]
Ts = [12]#,24,48]

output_dir = "../../output/"

# Read series
file_name = string("$(N)MS_$(T_max)_$(Sc)")
file_dir = "../../input/"
file = string(file_dir,file_name,".csv")
serie = readcsv(file, Float64)
serie = reshape(serie,N+1,T_max,Sc)

# Divide the series
ln_ret = serie[1:N,:,:]
rets = exp(ln_ret)-1 -dF.r_f

z = reshape(serie[N+1,:,:], T_max, Sc)

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

function memuse()
  pid = parse(Int,readall(`pidof julia`))
  return string(round(Int,parse(Int,readall(`ps -p $pid -o rss=`))/1024),"M")
end

function runMSDDP(dH, dM, Sc, rets_, states, ret_p)
  info("Memory use $(memuse())")
  ############ MSDDP ###########
  info("#### MSDDP ####")
  info("Training MSDDP...")
  @time LB, UB, LB_c, AQ, sp, list_α, list_β = sddp(dH, dM)

  info("Simulating MSDDP...")
  for se = 1:Sc
    x, x0, exp_ret = simulate(dH, dM, AQ, sp, rets_[:,:,se], states[:,se])
    ret_p[1,se] = x0[end]+sum(x[:,end])-1
  end
  info("Memory use $(memuse())")
  LB =0; UB =0; LB_c =0; AQ =0; sp =0; list_α =0; list_β =0;
  x  =0; x0  =0; exp_ret  =0;
end

function runOS(dF, dO, z_l, rets_, z, ret_p)
  ############ One Step ############
  info("#### One Step ####")
  info("Training One Step...")
  @time H_l, sp = OneStep.backward(dF, dO, z_l)

  info("Simulating One Step...")
  for se = 1:Sc
    x, x0 = OneStep.forward(dO, H_l, sp, z_l, z[:,se], rets_[:,:,se])
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
  @time u_l, Q_l = SDP.backward(dF, dS, z_l)
  # Use in one-step
  @time H_l, sp = OneStep.backward(dF, dO, z_l, Q_l)

  info("Simulating One Step FC...")
  for se = 1:Sc
    x, x0 = OneStep.forward(dO, H_l, sp, z_l, z[:,se], rets_[:,:,se])
    ret_p[3,se] = x0[end]+sum(x[:,end])-1.0
  end
  info("Memory use $(memuse())")
  u_l =0; Q_l =0; H_l =0; sp =0; x =0; x0 =0;
end

for dH.T in Ts
  dS.T = dO.T = dH.T
  rets_ = rets[:,1:dH.T,:]
  rets_p = zeros(3,length(γs),length(cs))
  file = string(output_dir,file_name,"_$(dH.T)_table.csv")
  for i_γ = 1:length(γs)
    dH.γ = dS.γ = dO.γ = γs[i_γ]
    for i_c = 1:length(cs)
      dH.c = dO.c = cs[i_c]
      ret_p = Array(Float64,3,Sc)
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
        rets_p[i,i_γ,i_c] = mean(ret_p[i,:])
      end
      open(file,"a") do x
        writecsv(x,hcat(dH.c, dH.γ,rets_p[:,i_γ,i_c]'))
      end
    end
  end
end
toc()

run(`/home/tas/woofy.sh 62491240 "simulateAR"`)
