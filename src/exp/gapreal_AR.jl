# Simulate AR multiple times to find the GAP for the "real" problem
addprocs(10)

using AR, Util
import SDP
using Distributions, Logging, JuMP, JLD
Logging.configure(level=Logging.DEBUG)

function evaluateGAP(dF, dS, z_l, rets, z, name,output_dir)
  # Evaluating the upper bound
  SampLB = 10
  UBs = Array(Float64,SampLB)

  for i=1:SampLB
    info("Memory use $(memuse())")
    info("Training SDP...")
    @time β = SDP.backward(dF, dS, z_l)
    p_s = ones(dS.S)*1/dS.S
    all = Array(Float64,dS.N+1, dS.T-1)
    Q_s, r_s = SDP.samplestp1(dS, dF, vec(β[2,:]), z_l, 0.0)
    Q = SDP.createmodel(dS, p_s, r_s, Q_s)
    status = solve(Q)
    if status ≠ :Optimal
      writeLP(Q,"prob.lp")
      error("Can't solve the problem status:",status)
    end
    UBs[i] = getobjectivevalue(Q)
  end
  β = SDP.backward(dF, dS, z_l)

  # Evaluating the lower bound
  rets_ = rets[:,1:dS.T,:]
  LBs = SharedArray(Float64,Sc)
  info("Simulating SDP...")
  @sync @parallel for se = 1:Sc
    W, all = SDP.forward(dS, dF, β, z_l, z[:,se], rets_[:,:,se])
    LBs[se] = W
  end

  # Evaluating real GAP(%)
  GAP = mean(UBs) - mean(LBs)
  GAP += 	quantile(Normal(), 0.99)*sqrt(var(UBs)/length(UBs) +var(LBs)/length(LBs))
  GAP /= mean(UBs)

  save("$(output_dir)realGapAR_$name.jld", "LBs", LBs, "UBs", UBs,"GAP",GAP)
  return GAP
end


include("parametersAR.jl")
# AR
dF = ARData(a_z, a_r, b_z, b_r, Σ, r_f)
dS = SDP.SDPData(N, T, L, S, α, γ)

# Split z
z_l = SDP.splitequaly(dS.L, z)

samps = collect(250:250:1000)
γs = [0.02,0.05,0.08,0.1,0.2]
GAPs = zeros(Float64,length(samps),length(γs))
for i_s = 1:length(samps)
  dS.S = samps[i_s]
  for i_γ = 1:length(γs)
    dS.γ = γs[i_γ]
    debug(dS)
    GAPs[i_s,i_γ] = evaluateGAP(dF, dS, z_l, rets, z, "$(dS.S)_$(dS.γ)",output_dir)
    info("GAP is $(GAPs[i_s,i_γ]) for $(dS.S) samples and γ $(dS.γ)")
  end
end
filename = string(output_dir,"realGap.csv")
writecsv(filename,[[0 ;γs]';[samps GAPs]])

run(`/home/tas/woofy.sh 62491240 "Finish $(@__FILE__) "`)
