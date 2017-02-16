using MSDDP, HMM_MSDDP, Simulate, Util
using Distributions, JLD
using Logging

import FFM
#addprocs(3)

srand(123)
Logging.configure(level=Logging.DEBUG)
include("parametersBM100.jl")
dH.GAPP = 0.2

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

γs = [0.05]#[0.005, 0.01,0.02]
cs = [0.01]#[0.005, 0.01,0.02]

file = string(output_dir,file_name,"_ret.csv")
dH.S_FB = 5 # One iteration one cut
dH.S_LB = 500 # fix amount of forwards
# For each risk level (γ)
@time for i_γ = 1:length(γs)
  dH.γ = γs[i_γ]
  # For each transactional cost (c)
  for i_c = 1:length(cs)
    dH.c = cs[i_c]
    debug(dH)
    file = string(output_dir,file_name)
    LB, UB, LB_c, AQ, sp, x_trial, u_trial, list_LB, List_UB,
      list_firstu = sddp(dH, dM;fastLBcal=false, file = string(output_dir,file_name) )

    γ_srt = string(dH.γ)[3:end]
    c_srt = string(dH.c)[3:end]
    save(string(output_dir,file_name,"_dataGAP.jld"), "l_LB", list_LB, "l_UB", list_UB, "l_firsu",list_firstu)
  end
end


run(`/home/tas/woofy.sh 62491240 "Finish $(@__FILE__) "`)
