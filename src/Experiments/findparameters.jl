using MSDDP
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

function generateseriesMS(T::Int64,N::Int64,S::Int64,T_l::Int64)
  σ = [0.002894	0.003532	0.00391	0.000115; 0.003532	0.004886	0.005712	0.000144; 0.00391	0.005712	0.007259	0.000163; 0.000115	0.000144	0.000163	0.0529]
  norm = MvNormal(σ)
  b_r = [ 0.0028 0.0049 0.0061]'
  b_z = 0.9700
  a_r  = [0.0053 0.0067 0.0067]'
  a_z  = 0.0000

  # Generate the series
  p = zeros(N+1,T,S)
  for s=1:S
    p[N+1,1,s] = 0
    for t=1:T
      sm = rand(norm)
      p[1:N,t,s] = a_r + b_r*p[N+1,t,s] + sm[1:N]
      if t < T
        p[N+1,t+1,s] = a_z + b_z*p[N+1,t,s] + sm[N+1]
      end
    end
  end

  # Teste the convergence of the series
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

  writecsv("../C++/input/$(N)MS_120_$(S).csv",p2[:])#hcat(p',z))
  #p2 = reshape(p[:],N+1,120,1000)
end

# Choose the number os samples for the LHS
function sampleslhs(dH::MSDDPData, file_name::AbstractString, file_dir::AbstractString)
  dH.K = 3
  last_std = 10000.0
  last_mean = 10000.0
  best_samp = 0
  for s=100:250:1200
    dH.S = s
    max_it = 10
    UBs = SharedArray(Float64,max_it)
    @sync @parallel for it=1:max_it #
      dH.N = N+1
      readall(`../C++/MS/HMM_MS /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $(dH.K) $(dH.N) $(dH.S) $(it)`)

      # HMM data
      file = string(file_dir,file_name,"_",it)
      dM = readHMMPara(file, dH)
      dM.r = dM.r[1:N,:,:] # removing the state z
      dH.N -= 1
      info("Train SDDP with $s LHS samples")
      @time LB, UB, AQ, sp, list_α, list_β, x_trial, u_trial, LB_c = SDDP(dH, dM)
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
function beststate(dH::MSDDPData, file_name::AbstractString, file_dir::AbstractString)
  last_ret = 0
  last_all = 0
  max_state = 7
  UBs = zeros(Float64,max_state)
  PVals = zeros(Float64,max_state)
  MRets = zeros(Float64,max_state)
  best_k = 0
  for k=1:7
    dH.K = k

    dH.N = N+1
    readall(`../C++/MS/HMM_MS /home/tas/Dropbox/PUC/PosDOC/ArtigoSDDP/Julia/C++ $file_name $(dH.K) $(dH.N) $(dH.S) $(k)`)

    # HMM data
    file = string(file_dir,file_name,"_",k)
    dM = readHMMPara(file, dH)
    dM.r = dM.r[1:N,:,:] # removing the state z
    dH.N -= 1
    info("Train SDDP with $k states")
    @time LB, UB, AQ, sp = SDDP(dH, dM)
    UBs[k] = UB

    #Simulate
    input_file = string("../C++/input/",file_name,".csv")
    r = readcsv(input_file,Float64)
    r = reshape(r,N+1,T_l,Sc)
    r = r[1:N,1:dH.T-1,:]
    r = exp(r)-1
    pk_r = readcsv(string(file,"_PK_r.csv"),Float64)'
    pk_r = reshape(pk_r,dH.K,T_l,Sc)
    pk_r = pk_r[:,1:dH.T-1,:]
    info("Simulating SDDP")
    ret = zeros(Float64,Sc)
    all = zeros(Float64,N+1,dH.T,Sc)
    for i=1:Sc
      x, x0, exp_ret = simulate(dH, dM, AQ, sp, r[:,:,i], pk_r[:,:,i] , x_ini_s[2:N+1], x_ini_s[1])
      ret[i] = x0[end]+sum(x[:,end])-1
      all[:,:,i] = vcat(x0',x)
    end
    MRets[k] = mean(ret)
    info("Mean return simulation $(mean(ret))")

    # Test t
    if k > 1
      testt = OneSampleTTest(ret,last_ret)
      PVals[k-1] = pvalue(testt)
      info("pvalue $(pvalue(testt)) with $k states")
      if DEBUG == true
        γ_srt = string(dH.γ)[3:end]
        c_srt = string(dH.c)[3:end]
        writecsv(string("./output/",file_name,"_$(γ_srt)$(c_srt)$(k)_all_.csv"),reshape(all,N+1,(dH.T)*Sc)')
        writecsv(string("./output/",file_name,"_$(γ_srt)$(c_srt)$(k)_ret.csv"),ret)
        writecsv(string("./output/",file_name,"_$(γ_srt)$(c_srt)_table.csv"),hcat(UBs,MRets,PVals))
      end
      if pvalue(testt) >= 0.05
        info("Fail to reject hypoteses with $k states. pvalue $(pvalue(testt))")
        dH.K -=1
        best_k = dH.K
        break
      end
    end
    last_ret = ret
    last_all = all
    # If couldn't stabilize the returns put -1
    if k == 7
      best_k = -1
    end
  end
  return best_k
end





DEBUG = true
srand(12345)
T_s = 240
N = 3
Sc = 1000
T_l = 120
#generateseriesMS(T_s,N,Sc,T_l)

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

γs = [0.001,0.003,0.006]
cs = [0.005,0.01,0.02]

file_name = "$(N)MS_120_$(Sc)"


c = cs[2]
γ = γs[2]

dH  = MSDDPData( N+1, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )
file_dir = "../C++/MS/output/"
#sampleslhs(dH, file_name, file_dir)

best_ks = zeros(Int64,length(γs),length(cs))
# For each risk level (γ)
for i_γ = 1:length(γs)
  γ = γs[i_γ]

  # For each transactional cost (c)
  for i_c = 1:length(cs)
    c = cs[i_c]
    info("Start testes with γ = $(γ) and c = $(c)")

    dH  = MSDDPData( N+1, T, K, S, α, x_ini_s[2:N+1], x_ini_s[1], c, M, γ, S_LB, S_FB, GAPP, Max_It, α_lB )
    file_dir = "../C++/MS/output/"

    dH.S = 500
    best_ks[i_γ,i_c] = beststate(dH, file_name, file_dir)
    writecsv(string("./output/",file_name,"_best_k.csv"),best_ks)
  end
end


run(`/home/tas/woofy.sh 62491240 "Finish FindParameters"`)
