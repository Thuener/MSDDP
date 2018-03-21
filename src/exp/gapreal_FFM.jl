#=
  Simulate MSDDP for markovian factors model
   multiple times to find the GAP for the "true" problem
=#

using MSDDP, HMM_MSDDP, FFM
using Distributions, Logging, JLD, CPLEX

function evaluateGAP(ms, maa, psddp, mk, Sc, rets_, states_, output_dir, name)
  # Evaluating the upper bound
  SampLB = 5 #TODO 10
  UBs = SharedArray(Float64, SampLB)
  LBs2mean = SharedArray(Float64, SampLB)
  LBs2var  = SharedArray(Float64, SampLB)
  list_firststage = SharedArray(Float64, nassets(ms)+1, SampLB)
  @sync @parallel for i=1:SampLB
    gc()
    Logging.configure(level=Logging.DEBUG)
    info("Memory use $(memuse())")
    info("Training MSDDP...")
    msddpm = MSDDPModel(maa, psddp, deepcopy(mk);
        lpsolver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=2))

    @time LB, UB, LB_c, x_trial, u_trial, list_LB,
        list_UB, list_firstu = solve(msddpm, param(msddpm); timelimit=3600*5)
    UBs[i] = UB
    list_firststage[:,i] =  list_firstu[:,end]

    gc()
    info("Evaluation of lower bound...")
    # Evaluation of lower bound
    LBs2 = Array(Float64,Sc)
    for se = 1:Sc
      x, x0, FO_forward, u_trial = MSDDP.forward!(msddpm, states_[:,se], rets_[:,:,se])
      LBs2[se] = FO_forward
    end
    LBs2mean[i] = mean(LBs2)
    LBs2var[i] = var(LBs2)
    save("$(output_dir)realGapFFM_$(name)_$i.jld", "UBs", UBs,"list_firststage",list_firststage,"LBs2",LBs2)
  end

  # Evaluating real GAP (%)
  GAP = Array(Float64, SampLB)
  for i = 1:SampLB
    GAP[i] = mean(UBs) - LBs2mean[i]
    GAP[i] += 	quantile(Normal(), 0.90)*sqrt(var(UBs)/length(UBs) +LBs2var[i]/Sc)
    GAP[i] = GAP[i]/mean(UBs)*100
  end

  save("$(output_dir)realGapFFM_$name.jld", "UBs", UBs, "GAP", GAP, "LBs2mean",LBs2mean, "LBs2var", LBs2var)
  return GAP
end


srand(123)
Logging.configure(level=Logging.DEBUG)
include("parametersBM100.jl")

ms = ModelSizes(T, N, K, S)
maa = MAAParameters(α, γ, c, x_ini, x0_ini, maxvl)
psddp = SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower; diff_upper = 0.005) #TODO 0.02

# TODO parameters used to run MSDDP with time limit
psddp.samplower =  0
psddp.fast_lower =  false
psddp.nit_before_lower = 1

output_dir = "../../output/outputFFM_P/"

# Read series
F =5 # number of factors
file_dir = "../../input/"
file_name = "5FF_BM100_Large"
file = string(file_dir,file_name,".csv")
data = readcsv(file, Float64)'

lnret = log(data+1)
df = FFM.evaluate(lnret[1:F,:],lnret[F+1:end,:])
mk, hmm = inithmm_ffm(lnret[1:F,:]', df, ms)

Sc = 1000
rets_f  = zeros(Float64, nassets(ms), nstages(ms))
states_f = Array(Int64, nstages(ms))
rets    = Array(Float64, nassets(ms), nstages(ms), Sc)
states  = Array(Int64, nstages(ms), Sc)

for se = 1:Sc
  simulatestates(ms, mk, states_f, rets_f)
  rets[:,:,se] = copy(rets_f)
  states[:,se] = copy(states_f)
end
debug(ms, maa, psddp)
msddpm = MSDDPModel(maa,
    psddp,
    deepcopy(mk);
    lpsolver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=2))

GAP = evaluateGAP(ms, maa, psddp, mk, Sc, rets, states, output_dir, "")
info("GAP is $(GAP) for $(nscen(ms)) samples")
