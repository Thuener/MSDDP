using MSDDP, HMM_MSDDP, AR, SDP
using Distributions, HypothesisTests, CPLEX
using Logging, ArgParse
import StatsBase


# Choose the number os samples for the LHS using SDP
function bestsamples_ttest(dS, output_dir, j)
  last_Ws = 0
  samps = collect(250:250:2500)
  len_samps = length(samps)
  MRets = zeros(Float64, len_samps)
  Stds = zeros(Float64, len_samps)
  PVals = zeros(Float64,len_samps)
  best_s = 0
  rets_ = rets[:,1:dS.T,:]

  pvaluelow = true
  for i = 1:length(samps)
    dS.S=samps[i]
    info("Testing SDP with $(dS.S) samples")
    β = backward(dF, dS, z_l)
    Ws = Array(Float64, Sc)
    for se = 1:Sc
      w, all = forward(dS, dF, β, z_l, vec(z[:,se]), rets_[:,:,se])
      Ws[se] = w
    end
    Stds[i] = std(Ws)
    MRets[i] = mean(Ws)
    info("Mean return simulation $(MRets[i])")

    # Test t
    if i > 1
      ttest = OneSampleTTest(Ws,last_Ws)
      PVals[i-1] = pvalue(ttest)
      info("pvalue $(pvalue(ttest)) with $(dS.S) samples")
      γ_srt = string(dS.γ)[3:end]
      if DEBUG
        writecsv(string(output_dir,file_name,"_$(γ_srt)$(dS.S)_ret$j.csv"), Ws)
      end
      writecsv(string(output_dir,file_name,"_$(γ_srt)_table_samp$j.csv"),hcat(MRets,Stds,PVals))
      if pvalue(ttest) < 0.05
        pvaluelow = true
      elseif pvaluelow
        info("Fail to reject hypoteses with $(dS.S) states. pvalue $(pvalue(ttest))")
        dS.S = samps[i-1]
        best_s = dS.S
        pvaluelow = false
        #break
      end
    end
    last_Ws = Ws
    # If couldn't stabilize the returns put -1
    if dS.S == samps[end]
      best_s = -1
    end
    gc()
  end
  return best_s
end

# Choose the number of sates for the HMM using SDP
function beststate_ttest(ms, maa, psddp, dF, dS, output_dir, j)
  max_state = 7

  UBs = zeros(Float64,max_state)
  PVals = zeros(Float64,max_state)
  MRets = zeros(Float64,2,max_state)
  Stds = zeros(Float64,2,max_state)
  best_k = 0
  rets_ = rets[:,1:nstages(ms),:]

  # Run SDP
  β = backward(dF, dS, z_l)
  Ws = Array(Float64, Sc)
  all_SDP = zeros(Float64,nassets(ms)+1,nstages(ms),Sc)
  x_ini = [1.0;zeros(N)]
  for se = 1:Sc
    w, all = forward(dS, dF, β, z_l, vec(z[:,se]), rets_[:,:,se])
    all_SDP[:,:,se] = hcat(x_ini,all)
    Ws[se] = w
  end
  MRets[2,1] = mean(Ws)
  Stds[2,1] = std(Ws)
  γ_srt = string(maa.γ)[3:end]
  writecsv(string(output_dir,file_name,"_$(γ_srt)_SDP_all$j.csv"),reshape(all_SDP,nassets(ms)+1,(nstages(ms))*Sc)')
  DEBUG && writecsv(string(output_dir,file_name,"_$(γ_srt)_w$j.csv"),Ws)

  for k=1:max_state
    setnstates!(ms,k)

    # HMM data
    #dM, model = inithmm_z(reshape(ln_ret, nassets(ms) +1, T_hmm*Sc)', dH, T_hmm, Sc)
    z_slothmm = splitequaly(k, z)
    v_hmm = dF.Σ[nassets(ms)+1,nassets(ms)+1]*ones(k)
    mk, hmm = inithmm_ar(z[T_max-T_hmm+1:T_max,:], dF, ms, T_hmm, Sc, z_slothmm, v_hmm)

    info("Train SDDP with $k states")
    model = MSDDPModel(ms, maa, psddp, mk, CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=2))
    @time LB, UB, LB_c, = solve(model, param(model))
    UBs[k] = UB

    #Simulate
    info("Simulating SDDP")
    ret_c = zeros(Float64,Sc)
    count_states = zeros(Int64,nstates(ms))
    all_c = zeros(Float64,nassets(ms)+1,nstages(ms),Sc)
    for s=1:Sc
      #states = predict(model,ln_ret[:,1:nstages(ms)-1,s]')
      states = predict(hmm, z[1:nstages(ms)-1,s])
      count_states .+= StatsBase.counts(states,1:nstates(ms))
      x, x0, exp_ret = simulate(model, rets_[:,:,s], states)
      all_c[:,:,s] = vcat(x0',x)
      ret_c[s] = x0[end]+sum(x[:,end])
    end
    debug("count_states =", count_states)
    MRets[1,k] = mean(ret_c)
    Stds[1,k] = std(Ws-ret_c)
    info("Mean return simulation $(MRets[k])")


    # Test t
    ttest = OneSampleTTest(Ws,ret_c)
    PVals[k] = pvalue(ttest, tail=:left)
    info("pvalue $(pvalue(ttest)) with $k states")
    if DEBUG
      writecsv(string(output_dir,file_name,"_$(γ_srt)$(k)_all$j.csv"),reshape(all_c,nassets(ms)+1,(nstages(ms))*Sc)')
      writecsv(string(output_dir,file_name,"_$(γ_srt)$(k)_ret$j.csv"),ret_c)
    end
    writecsv(string(output_dir,file_name,"_$(γ_srt)_table_states$j.csv"),hcat(UBs,MRets',Stds',PVals))
    if pvalue(ttest) >= 0.05
      info("Fail to reject hypoteses with $k states. pvalue $(pvalue(ttest))")
      best_k = nstates(ms)
      #break
    end

    # If couldn't stabilize the returns put -1
    if k == max_state
      best_k = -1
    end
  end
  return best_k
end



function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table s begin
    "--debug", "-d"
        help = "Show debug messanges"
        action = :store_true
    "--samp", "-S"
        help = "To selecting samples of the LHS"
        action = :store_true
    "--stat", "-K"
        help = "To selecting the number of states of the Markov model"
        action = :store_true
  end

  return parse_args(s)
end


# Start of the scipt
args = parse_commandline()

if args["debug"]
  Logging.configure(level=Logging.DEBUG)
  DEBUG = true
else
  Logging.configure(level=Logging.INFO)
end

srand(111)
T_max = 240
N = 3
Sc = 1000
T_hmm = 120

# Parameters for AR
Σ = [0.002894	0.003532	0.00391	-0.000115; 0.003532	0.004886	0.005712	-0.000144; 0.00391	0.005712	0.007259	-0.000163; -0.000115	-0.000144	-0.000163	0.0529]
b_r = [ 0.0028; 0.0049; 0.0061]
b_z = [0.9700]
a_r  = [0.0053; 0.0067; 0.0072]
a_z  = [0.0000]
r_f = 0.00042

dF = ARData(a_z, a_r, b_z, b_r, Σ, r_f)

# Parameters for MSDDPData
N = 3
T = 13
K = 4
S = 750
α = 0.9
x0_ini = 1.0
x_ini = zeros(N)
c = 0.00
maxvl = 9999999
γ = 0.012
samplower = 300
samplower_inc = 100
nit_before_lower = 100
gap = 0.05
max_it = 100
α_lower = 0.9

γs = [0.02,0.05,0.08,0.1,0.2]
cs = [0.005,0.01,0.02]

# Parameters for SDP
L = 1000
dS = SDPData(N, T, L, S, α, γ)

# Read series
file_name = string("$(N)MS_$(T_max)_$(Sc)")
file_dir = "../../input/"
file = string(file_dir,file_name,".csv")
serie = readcsv(file, Float64)
serie = reshape(serie,N+1,T_max,Sc)

# Generate the series ρ and z
#serie_ = series(dF, Sc, T_hmm, T_max)

# Divide the series
ln_ret = serie[1:N,:,:]
rets = exp(ln_ret)-1 -dF.r_f
z = reshape(serie[N+1,:,:], T_max, Sc)

# Split z
z_l = splitequaly(dS.L, z)

if args["samp"]
  bests_sam = zeros(Int64, length(γs))
  for j = 1:1
    # For each risk level (γ)
    for i_γ = 1:length(γs)
      dS.γ = γs[i_γ]
      info("Start testes with γ = $(dS.γ)")
      output_dir = "../../output/"

      bests_sam[i_γ] = bestsamples_ttest(dS, output_dir, j)
      writecsv(string(output_dir, file_name, "_best_S$(j).csv"),bests_sam)
    end
  end
end

if args["stat"]
  best_ks = zeros(Int64,length(γs))
  for j = 1:1
    # For each risk level (γ)
    for i_γ = 1:length(γs)
      γ = γs[i_γ]
      dS.γ = γ
      info("Start testes with γ = $(dS.γ)")

      ms = ModelSizes(T, N, K, S)
      maa = MAAParameters(α, γ, c, x_ini, x0_ini, maxvl)
      psddp = SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower)

      output_dir = "../../output/"
      best_ks[i_γ] = beststate_ttest(ms, maa, psddp, dF, dS, output_dir, j)
      writecsv(string(output_dir, file_name, "_best_K$(j).csv"),best_ks)
    end
  end
end

run(`/home/tas/woofy.sh 62491240 "Finish $(@__FILE__) "`)
