using MSDDP, HMM_MSDDP
using Distributions, JLD, CPLEX
using Logging

import FFM
#addprocs(3)

srand(123)
Logging.configure(level=Logging.DEBUG)
include("parametersBM100.jl")

ms = ModelSizes(T, N, K, S)
maa = MAAParameters(α, γ, c, x_ini, x0_ini, maxvl)
psddp = SDDPParameters(max_it, samplower, samplower_inc, nit_before_lower, gap, α_lower; diff_upper = 0.02)

output_dir = "../../output/outputFFM/"

# Read series
F =5 # number of factors
file_dir = "../../input/"
file_name = "5FF_BM100_Large"
file = string(file_dir,file_name,".csv")
data = readcsv(file, Float64)'

lnret = log(data+1)
df = FFM.evaluate(lnret[1:F,:],lnret[F+1:end,:])
mk, hmm = inithmm_ffm(lnret[1:F,:]', df, ms)

γs = [0.05]#[0.005, 0.01,0.02]
cs = [0.01]#[0.005, 0.01,0.02]

file = string(output_dir,file_name,"_ret.csv")
psddp.nit_before_lower = 5 # One iteration one cut
psddp.samplower = 500 # fix amount of forwards
psddp.samplower_inc = 0
psddp.fast_lower =  false
psddp.file = string(output_dir,file_name)

msddpm = MSDDPModel(maa,
    psddp,
    deepcopy(mk);
    lpsolver = CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_LPMETHOD=2))

# For each risk level (γ)
@time for i_γ = 1:length(γs)
  setγ!(msddpm, γs[i_γ])
  # For each transactional cost (c)
  for i_c = 1:length(cs)
    settranscost!(msddpm, cs[i_c])
    reset!(msddpm)
    file = string(output_dir,file_name)
    LB, UB, LB_c, x_trial, u_trial, list_LB,
        list_UB, list_firstu = solve(msddpm, param(msddpm))

    γ_srt = string(γs[i_γ])[3:end]
    c_srt = string(cs[i_c])[3:end]
    save(string(output_dir,file_name,"_dataGAP.jld"), "l_LB", list_LB, "l_UB", list_UB, "l_firsu",list_firstu)
  end
end


run(`/home/tas/woofy.sh 62491240 "Finish $(@__FILE__) "`)
