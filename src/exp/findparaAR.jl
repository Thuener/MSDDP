using HMM_MSDDP, AR
using Distributions, HypothesisTests
using Logging, ArgParse


# Choose the number os samples for the LHS
# using standard deviation stabilization of the UB
function sampleslhs_stabUB(dH::MSDDPData, ln_ret::Array{Float64,3}, T_l::Int64, Sc::Int64)
  last_std = 10000.0
  last_mean = 10000.0
  best_samp = 0
  for s=250:250:1000
    dH.S = s
    max_it = 10
    UBs = SharedArray(Float64,max_it)
    @sync @parallel for it=1:max_it
      #dM, model = inithmm_z(reshape(ln_ret,dH.N+1,T_l*Sc)', dH, T_l, Sc)
      dM, model, y = inithmm_ar(ln_ret, dF, dH, T_l, Sc)

      # HMM data
      dM.r = dM.r[1:dH.N,:,:] # removing the state z

      info("Train SDDP with $s LHS samples")
      @time LB, UB, LB_c, AQ, sp, x_trial, u_trial, LB_c = sddp(dH, dM)
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

# Choose the number os samples for the LHS
# when it stops to change the allocation (t-test)
function bestsamples_ttest(output_dir::AbstractString, dH::MSDDPData, dF::ARData, ln_ret::Array{Float64,3}, T_l::Int64, Sc::Int64)
  last_ret_c = 0
  last_all_c = 0
  samps = collect(250:250:1000)
  len_samps = length(samps)
  UBs = zeros(Float64, len_samps)
  PVals = zeros(Float64, len_samps)
  MRets = zeros(Float64, len_samps)
  best_s = 0
  z = ln_ret[end,:,:]
  ret_e = exp(ln_ret[1:dH.N,:,:])-1 -dF.r_f

  for i = 1:length(samps)
    dH.S=samps[i]
    # HMM data
    #dM, model = inithmm_z(reshape(ln_ret, dH.N +1, T_l*Sc)', dH, T_l, Sc)
    dM, model, y = inithmm_ar(z, dF, dH, T_l, Sc)

    info("Train SDDP with $(dH.S) samples")
    @time LB, UB, LB_c, AQ, sp = sddp(dH, dM)
    UBs[i] = UB

    #Simulate
    info("Simulating SDDP")
    ret_c = zeros(Float64,Sc)
    all_c = zeros(Float64,dH.N+1,dH.T,Sc)
    for s=1:Sc
      #states = predict(model,ln_ret[:,1:dH.T-1,s]')
      states = predict(model,y[:,1:dH.T-1,s]')
      x, x0, exp_ret = simulate(dH, dM, AQ, sp, ret_e[:,1:dH.T-1,s], states)
      ret_c[s] = x0[end]+sum(x[:,end])
      all_c[:,:,s] = vcat(x0',x)
    end
    MRets[i] = mean(ret_c)
    info("Mean return simulation $(MRets[i])")

    # Test t
    if i > 1
      ttest = OneSampleTTest(ret_c,last_ret_c)
      PVals[i-1] = pvalue(ttest)
      info("pvalue $(pvalue(ttest)) with $(dH.S) samples")
      γ_srt = string(dH.γ)[3:end]
      c_srt = string(dH.c)[3:end]
      if DEBUG
        writecsv(string(output_dir,file_name,"_$(γ_srt)$(c_srt)$(dH.S)_all_.csv"),reshape(all_c,dH.N+1,(dH.T)*Sc)')
        writecsv(string(output_dir,file_name,"_$(γ_srt)$(c_srt)$(dH.S)_ret.csv"),ret_c)
      end
      writecsv(string(output_dir,file_name,"_$(γ_srt)$(c_srt)_table_samp.csv"),hcat(UBs,MRets,PVals))
      if pvalue(ttest) >= 0.05
        info("Fail to reject hypoteses with $(dH.S) states. pvalue $(pvalue(ttest))")
        dH.S = samps[i-1]
        best_s = dH.S
        break
      end
    end
    last_ret_c = ret_c
    last_all_c = all_c
    # If couldn't stabilize the returns put -1
    if dH.S == samps[end]
      best_s = -1
    end
  end
  return best_s
end

# Choose the number of sates for the HMM
# when it stops to change the allocation (UnequalVarianceTTest)
function beststate_equal(output_dir::AbstractString, dH::MSDDPData, dF::ARData, ln_ret::Array{Float64,3}, T_l::Int64, Sc::Int64)
  ret_hmm_p_last = 0
  max_state = 20
  PVals = zeros(Float64, max_state)
  MRets = zeros(Float64, 2, max_state)
  best_k = 0
  ret = exp(ln_ret)-1 - dF.r_f
  for k=1:max_state
    dH.K = k
    info("Simulating with $k states")
    # HMM data
    #dM, model = inithmm_z(reshape(ln_ret, dH.N +1, T_l*Sc)', dH, T_l, Sc)
    dM, model, y = inithmm_ar(ln_ret, dF, dH, T_l, Sc)

    ret_hmm_p = zeros(Sc)
    ret_ar_p  = zeros(Sc)
    dH.c = 0
    for se = 1:Sc
      zs, states = model[:sample](n_samples=T_l)
      norm = MvNormal(dF.Σ)
    	#n_comp = 2 # only uses the second componet of the HMM (z_t)
      ret_hmm = zeros(dH.N, T_l)
      zs[1] = dF.a_z[1]
      for t = 1:T_l
        sm = rand(norm)
        ρ = dF.a_r + dF.b_r*zs[t] + sm[1:dH.N]
  			ret_hmm[:,t] = exp(ρ)-1
        ret_hmm[:,t] -= dF.r_f
      end

      # Simulate with HMM serie
      #ret_hmm = ones(dH.N,1)*maximum(ret_hmm,1)
      all_hmm = simulate_percport(dH, ret_hmm, [1.0;zeros(N)], [0;ones(dH.N)/(dH.N)])
      ret_hmm_p[se] = sum(all_hmm[:,2:end])


      # Simulate with AR series
      #ret_ar = series_assets(dF, 240, 1, T_l)
      ret_ar = ret[1:dH.N,:,se]
      ret_ar = reshape(ret_ar,dH.N,T_l)

      #ret_ar = ones(dH.N,1)*maximum(ret_ar,1)
      all_ar = simulate_percport(dH, ret_ar, [1.0;zeros(N)], [0;ones(dH.N)/(dH.N)])
      ret_ar_p[se] = sum(all_ar[:,2:end])
    end
    if k > 1
      ttest = UnequalVarianceTTest(ret_hmm_p,ret_hmm_p_last)
      PVals[k-1] = pvalue(ttest)
      info("hmm_pvalue $(pvalue(ttest)) with $k states")
      if pvalue(ttest) >= 0.05
        info("Fail to reject hypoteses with $k states. pvalue $(pvalue(ttest))")
      end
    end
    ret_hmm_p_last = ret_hmm_p
    MRets[1,k] = mean(ret_hmm_p)
    MRets[2,k] = mean(ret_ar_p)
    info("Mean return simulation HMM $(MRets[1,k]) and AR $(MRets[2,k])")

    # Test t
    ttest = ApproximateTwoSampleKSTest(ret_hmm_p, ret_ar_p)
    PVals[k] = pvalue(ttest,tail=:left)
    info("pvalue $(pvalue(ttest)) $(pvalue(ttest,tail=:left)) $(pvalue(ttest,tail=:right)) with $k states ")
    if DEBUG
      γ_srt = string(dH.γ)[3:end]
      c_srt = string(dH.c)[3:end]
      writecsv(string(output_dir,file_name,"_$(k)_hmm.csv"),ret_hmm_p)
      writecsv(string(output_dir,file_name,"_$(k)_ar.csv"),ret_ar_p)
    end
    writecsv(string(output_dir,file_name,"_table_states.csv"),hcat(MRets',PVals))
    if pvalue(ttest,tail=:left) >= 0.05
      info("Fail to reject hypoteses with $k states. pvalue $(pvalue(ttest,tail=:left))")
      best_k = dH.K
      break
    end
    # If couldn't stabilize the returns put -1
    if k == max_state
      best_k = -1
    end
  end
  return best_k
end

# Choose the number of sates for the HMM
# when it stops to change the allocation (t-test)
function beststate_ttest(output_dir::AbstractString, dH::MSDDPData, dF::ARData, ln_ret::Array{Float64,3}, T_l::Int64, Sc::Int64)
  last_ret_c = 0
  last_all_c = 0
  max_state = 7

  UBs = zeros(Float64,max_state)
  PVals = zeros(Float64,max_state)
  MRets = zeros(Float64,max_state)
  best_k = 0
  ret_e = exp(ln_ret)-1 - dF.r_f
  for k=1:max_state
    dH.K = k

    # HMM data
    #dM, model = inithmm_z(reshape(ln_ret, dH.N +1, T_l*Sc)', dH, T_l, Sc)
    dM, model, y = inithmm_ar(ln_ret, dF, dH, T_l, Sc)

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
      ret_c[s] = x0[end]+sum(x[:,end])
      all_c[:,:,s] = vcat(x0',x)
    end
    MRets[k] = mean(ret_c)
    info("Mean return simulation $(MRets[k])")

    # Test t
    if k > 1
      ttest = OneSampleTTest(ret_c,last_ret_c)
      PVals[k-1] = pvalue(ttest)
      info("pvalue $(pvalue(ttest)) with $k states")
      γ_srt = string(dH.γ)[3:end]
      c_srt = string(dH.c)[3:end]
      if DEBUG
        writecsv(string(output_dir,file_name,"_$(γ_srt)$(c_srt)$(k)_all.csv"),reshape(all_c,dH.N+1,(dH.T)*Sc)')
        writecsv(string(output_dir,file_name,"_$(γ_srt)$(c_srt)$(k)_ret.csv"),ret_c)
      end
      writecsv(string(output_dir,file_name,"_$(γ_srt)$(c_srt)_table_states.csv"),hcat(UBs,MRets,PVals))
      if pvalue(ttest) >= 0.05
        info("Fail to reject hypoteses with $k states. pvalue $(pvalue(ttest))")
        dH.K -=1
        best_k = dH.K
        break
      end
    end

    last_ret_c = ret_c
    last_all_c = all_c
    # If couldn't stabilize the returns put -1
    if k == max_state
      best_k = -1
    end
  end
  return best_k
end


function slidingwindow(data::Array{Float64,2}, k::Int64)
  Sc = size(data,2)
  T_l = 120
  T = 240
  sum = 0
  its = 10
  comp = 2
  train = reshape(series_zs(dF, T, Sc, T_l), comp, Sc*(T_l-1))'
  lst = fill(T_l-1, Sc)
  model = train_hmm(train, k, lst)
  for i = 1:its
    test  = reshape(series_zs(dF, T, Sc, T_l), comp, Sc*(T_l-1))'
    logll = score(model, test, lst)
    sum += logll
  end
  return sum/its
end

# Choose the number of sates for the HMM
function beststate_slidingwindow(z::Array{Float64,2}, Sc::Int64)
  max_state = 10
  best_k = 0
  max_logll = -Inf
  for k=1:max_state
    logll = slidingwindow(z, k)
    debug("With $k states, logll $logll")
    if logll > max_logll
      max_logll = logll
      best_k = k
      debug("New best with $k states, logll $max_logll ")
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

srand(123)
T_s = 240
N = 3
Sc = 1000
T_l = 120

# AR
Σ = [0.002894	0.003532	0.00391	-0.000115; 0.003532	0.004886	0.005712	-0.000144; 0.00391	0.005712	0.007259	-0.000163; -0.000115	-0.000144	-0.000163	0.0529]
b_r = [ 0.0028; 0.0049; 0.0061]
b_z = [0.9700]
a_r  = [0.0053; 0.0067; 0.0072]
a_z  = [0.0000]
r_f = 0.00042

dF = ARData(a_z, a_r, b_z, b_r, Σ, r_f)

#Parameters
N = 3
T = 13
K = 4
S = 500
α = 0.9
W_ini = 1.0
x_ini_s = [W_ini;zeros(N)]
c = 0.005
M = 9999999
γ = 0.012
S_LB = 300
S_LB_inc = 100
S_FB = 100
GAPP = 1
Max_It = 100
α_lB = 0.9

γs = [0.05,0.1, 0.2,0.3]
cs = [0.005,0.01,0.02]

#series(dF, Sc, T_l, T_s)
#p2 = reshape(p[:],N+1,120,1000)
file_name = string("$(N)MS_120_$(Sc)",".csv")
file_dir = "../../input/"


file = string(file_dir,file_name)
ln_ret = readcsv(file, Float64)
ln_ret = reshape(ln_ret,N+1,T_l,Sc)

if args["stat"]
  best_ks = zeros(Int64,length(γs),length(cs))
  # For each risk level (γ)
  for i_γ = 1:length(γs)
    γ = γs[i_γ]

    # For each transactional cost (c)
    for i_c = 1:length(cs)
      c = cs[i_c]
      info("Start testes with γ = $(γ) and c = $(c)")

      dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], W_ini, c, M, γ,
                      S_LB, S_LB_inc, S_FB, GAPP, Max_It, α_lB )
      output_dir = "../../output2/"

      best_ks[i_γ,i_c] = beststate_ttest(output_dir, dH, dF, ln_ret, T_l, Sc)
      writecsv(string(output_dir, file_name, "_best_K.csv"),best_ks)
    end
  end
end

if args["samp"]
  bests_sam = zeros(Int64, length(γs), length(cs))
  # For each risk level (γ)
  for i_γ = 1:length(γs)
    γ = γs[i_γ]

    # For each transactional cost (c)
    for i_c = 1:length(cs)
      c = cs[i_c]
      info("Start testes with γ = $(γ) and c = $(c)")

      dH  = MSDDPData( N, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], W_ini, c, M, γ,
                      S_LB, S_LB_inc, S_FB, GAPP, Max_It, α_lB )
      output_dir = "../../output/"

      bests_sam[i_γ,i_c] = bestsamples_ttest(output_dir, dH, dF, ln_ret, T_l, Sc)
      writecsv(string(output_dir, file_name, "_best_S.csv"),bests_sam)
    end
  end
end

run(`/home/tas/woofy.sh 62491240 "Finish $(@__FILE__) "`)
