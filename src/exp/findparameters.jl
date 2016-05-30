using HMM_MSDDP
using Distributions
using HypothesisTests
using Logging
Logging.configure(level=Logging.DEBUG)

#=
function debug(msg)
  println("DEBUG: ",msg)
end

function Base.info(msg)
  println("INFO: ",msg)
end
=#

function generateseriesMS(dF::Factors,T::Int64,N::Int64,S::Int64,T_l::Int64)
  norm = MvNormal(dF.Σ)

  # Generate the series
  p = zeros(N+1,T,S)
  for s=1:S
    p[N+1,1,s] = dF.a_z[1]
    for t=1:T
      sm = rand(norm)
      p[1:N,t,s] = dF.a_r + dF.b_r*p[N+1,t,s] + sm[1:N]
      if t < T
        p[N+1,t+1,s] = dF.a_z[1] + dF.b_z[1]*p[N+1,t,s] + sm[N+1]
      end
    end
  end

  # Test the convergence of the series
  var_T = var(p[N+1,T,:])
  info("var_T $var_T")
  for t=1:T
    var_t  = var(p[N+1,t,:])
    info("var_t $t $var_t")
    if abs(var_t - var_T) < 0.009
      break
    end
  end
  p2 = p[:,T_l+1:T,:]

  writecsv("../../input/$(N)MS_120_$(S).csv",p2[:])#hcat(p',z))
  #p2 = reshape(p[:],N+1,120,1000)
end

# Choose the number os samples for the LHS
function sampleslhs(dH::MSDDPData, ln_ret::Array{Float64,3}, T_l::Int64, Sc::Int64)
  dH.K = 3
  last_std = 10000.0
  last_mean = 10000.0
  best_samp = 0
  for s=100:200:700
    dH.S = s
    max_it = 10
    UBs = SharedArray(Float64,max_it)
    @sync @parallel for it=1:max_it
      #dM, model = inithmm(reshape(ln_ret,dH.N+1,T_l*Sc)', dH, T_l, Sc)
      dM, model, y = inithmm_onefactor(ln_ret, dF, dH, T_l, Sc)

      # HMM data
      dM.r = dM.r[1:dH.N,:,:] # removing the state z

      info("Train SDDP with $s LHS samples")
      @time LB, UB, LB_c, AQ, list_α, list_β, x_trial, u_trial, LB_c = sddp(dH, dM)
      UBs[it] = UB
    end
    curr_std = sqrt(var(UBs))
    curr_mean = mean(UBs)
    info("Test with $s samples, STDs $(last_std) $(curr_std) GAP_STD $(abs(last_std - curr_std)/curr_std)")
    info("Test with $s samples, MEANs $(last_mean) $(curr_mean) GAP_MEAN $(abs(last_mean - curr_mean)/curr_mean)")
    if abs(curr_std - last_std)/curr_std < 1e-2
      info("Stabilization with $s samples")
      dH.S = s-100
      best_samp = dH.S
      break
    end

    last_std = curr_std
    last_mean = curr_mean
  end
  return best_samp
end

# Choose the number of sates for the HMM
function beststate(dH::MSDDPData, dF::Factors, ln_ret::Array{Float64,3}, T_l::Int64, Sc::Int64)
  last_ret_c = 0
  last_all_c = 0
  max_state = 7
  output_dir = "../../output/"
  UBs = zeros(Float64,max_state)
  PVals = zeros(Float64,max_state)
  MRets = zeros(Float64,max_state)
  best_k = 0
  ret_e = exp(ln_ret)-1
  for k=1:7
    dH.K = k

    # HMM data
    #dM, model = inithmm(reshape(ln_ret, dH.N +1, T_l*Sc)', dH, T_l, Sc)
    dM, model, y = inithmm_onefactor(ln_ret, dF, dH, T_l, Sc)

    info("Train SDDP with $k states")
    @time LB, UB, LB_c, AQ, sp = sddp(dH, dM)
    UBs[k] = UB

    #Simulate
    info("Simulating SDDP")
    ret_c = zeros(Float64,Sc)
    all_c = zeros(Float64,dH.N+1,dH.T,Sc)
    for s=1:Sc
      #states = predict(model,ln_ret[:,1:dH.T-1,s]')
      states = predict(model,y[:,1:dH.T-1,s]')
      x, x0, exp_ret = simulate(dH, dM, AQ, sp, ret_e[1:dH.N,1:dH.T-1,s], states)
      ret_c[s] = x0[end]+sum(x[:,end])-1
      all_c[:,:,s] = vcat(x0',x)
    end
    MRets[k] = mean(ret_c)
    info("Mean return simulation $(MRets[k])")

    # Test t
    if k > 1
      testt = OneSampleTTest(ret_c,last_ret_c)
      PVals[k-1] = pvalue(testt)
      info("pvalue $(pvalue(testt)) with $k states")
      if DEBUG == true
        γ_srt = string(dH.γ)[3:end]
        c_srt = string(dH.c)[3:end]
        writecsv(string(output_dir,file_name,"_$(γ_srt)$(c_srt)$(k)_all_.csv"),reshape(all_c,dH.N+1,(dH.T)*Sc)')
        writecsv(string(output_dir,file_name,"_$(γ_srt)$(c_srt)$(k)_ret.csv"),ret_c)
        writecsv(string(output_dir,file_name,"_$(γ_srt)$(c_srt)_table.csv"),hcat(UBs,MRets,PVals))
      end
      if pvalue(testt) >= 0.05
        info("Fail to reject hypoteses with $k states. pvalue $(pvalue(testt))")
        dH.K -=1
        best_k = dH.K
        break
      end
    end
    last_ret_c = ret_c
    last_all_c = all_c
    # If couldn't stabilize the returns put -1
    if k == 7
      best_k = -1
    end
  end
  return best_k
end



DEBUG = true
srand(123)
T_s = 240
N = 3
Sc = 1000
T_l = 120
# Factors
Σ = [0.002894	0.003532	0.00391	-0.000115; 0.003532	0.004886	0.005712	-0.000144; 0.00391	0.005712	0.007259	-0.000163; -0.000115	-0.000144	-0.000163	0.0529]
b_r = [ 0.0028; 0.0049; 0.0061]
b_z = [0.9700]
a_r  = [0.0053; 0.0067; 0.0072]
a_z  = [0.0000]
r_f = 1.00042

dF = Factors(a_z, a_r, b_z, b_r, Σ, r_f)
#generateseriesMS(dF, T_s, N, Sc, T_l)# Only one

#Parameters
N = 3
T = 13
K = 4
S = 500
α = 0.9
x_ini_s = [1.0;zeros(N)]
c = 0.005
M = 9999999
γ = 0.012
S_LB = 300
S_FB = 100
GAPP = 1
Max_It = 100
α_lB = 0.9

γs = [0.02,0.05,0.08]
cs = [0.005,0.01,0.02]

file_name = string("$(N)MS_120_$(Sc)",".csv")
file_dir = "../../input/"


file = string(file_dir,file_name)
ret = readcsv(file, Float64)
ret = reshape(ret,N+1,T_l,Sc)

c = cs[2]
γ = γs[2]

#dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )
#sampleslhs(dH, ret, T_l, Sc)

best_ks = zeros(Int64,length(γs),length(cs))
# For each risk level (γ)
for i_γ = 1:length(γs)
  γ = γs[i_γ]

  # For each transactional cost (c)
  for i_c = 1:length(cs)
    c = cs[i_c]
    info("Start testes with γ = $(γ) and c = $(c)")

    dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )
    output_dir = "../../output/"

    dH.S = 500
    best_ks[i_γ,i_c] = beststate(dH, dF, ret, T_l, Sc)
    writecsv(string(output_dir,file_name,"_best_k.csv"),best_ks)
  end
end


run(`/home/tas/woofy.sh 62491240 "Finish FindParameters"`)
