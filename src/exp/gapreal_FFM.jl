#=
  Simulate MSDDP for markovian factors model
   multiple times to find the GAP for the "real" problem
=#
using MSDDP, HMM_MSDDP, FFM, Util
using Distributions, Logging, JuMP, JLD
Logging.configure(level=Logging.DEBUG)

function evaluateGAP(dH, dM, Sc, rets_, states_, output_dir, name)
  # Evaluating the upper bound
  SampLB = 10
  UBs = Array(Float64,SampLB)
  AQ = sp =0
  list_firststage = Array(Float64, dH.N+1, SampLB)
  for i=1:SampLB
    info("Memory use $(memuse())")
    info("Training MSDDP...")
    @time LB, UB, LB_c, AQ, sp, x_trial, u_trial, list_LB,
      list_UB, list_firstu = MSDDP.sddp(dH, dM;stabUB=0.05)
    UBs[i] = UB
    list_firststage[:,i] =  list_firstu[:,end]
    writecsv("$(output_dir)firststage_$name.csv", list_firststage)
  end

  # Evaluating the lower bound
  LBs = Array(Float64,Sc)
  info("Simulating MSDDP...")
  for se = 1:Sc
    x, x0 = MSDDP.forward(dH, dM, AQ, sp, states[:,se], rets_[:,:,se])
    LBs[se] = x0[end]+sum(x[:,end])-1
  end

  # Evaluating real GAP
  GAP = mean(UBs) - mean(LBs)
  GAP += 	quantile(Normal(), 0.95)*sqrt(var(UBs)/length(UBs) +var(LBs)/length(LBs))

  save("$(output_dir)realGapFFM_$name.jld", "LBs", LBs, "UBs", UBs,"GAP",GAP)
  return GAP
end


srand(123)
include("parametersBM100.jl")

output_dir = "../../output/outputFFM/"

# Read series
F =5 # number of factors
file_dir = "../../input/"
file_name = "5FF_BM100_Large"
file = string(file_dir,file_name,".csv")
data = readcsv(file, Float64)'

lnret = log(data+1)
dF = FFM.evaluate(lnret[1:F,:],lnret[F+1:end,:])
dM, model = inithmm_ffm(lnret[1:F,:]', dF, dH)

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

GAP = evaluateGAP(dH, dM, Sc, rets, states, output_dir, "")
info("GAP is $(GAPs[i]) for $(dH.S) samples")
