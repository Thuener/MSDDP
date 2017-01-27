# Simulate AR multiple times to find the GAP for the "real" problem
using AR, Util
import SDP
using Distributions, Logging, JuMP, JLD
Logging.configure(level=Logging.DEBUG)

function evaluateGAP(dF, dS, z_l, rets, z, name,output_dir)
  # Evaluating the upper bound
  SampLB = 10
  UBs = SharedArray(Float64,SampLB)
  β = 0
  @sync @parallel for i=1:SampLB
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
    UBs[i] = getobjectivevalue(Q)-1
  end

  # Evaluating the lower bound
  rets_ = rets[:,1:dS.T,:]
  LBs = Array(Float64,Sc)
  for se = 1:Sc
    W, all = SDP.forward(dS, dF, β, z_l, z[:,se], rets_[:,:,se])
    LBs[se] = W-1.0
  end

  # Evaluating real GAP
  GAP = mean(UBs) - mean(LBs)
  GAP += 	quantile(Normal(), 0.999)*sqrt(var(UBs)/length(UBs) +var(LBs)/length(LBs))

  save("$(output_dir)realGapAR_$name.jld", "LBs", LBs, "UBs", UBs,"GAP",GAP)
  return GAP
end


include("parametersAR.jl")
# AR
dF = ARData(a_z, a_r, b_z, b_r, Σ, r_f)
dS = SDP.SDPData(N, T, L, S, α, γ)

# Split z
z_l = SDP.splitequaly(dS.L, z)

samps = vcat(collect(50:50:200),collect(250:250:1000))
GAPs = zeros(Float64,length(samps))
for i = 1:length(samps)
  dS.S=samps[i]
  GAPs[i] = evaluateGAP(dF, dS, z_l, rets, z, "$(dS.S)",output_dir)
  info("GAP is $(GAPs[i]) for $(dS.S) samples")
end
filename = string(output_dir,"realGap.csv")
writecsv(filename,hcat(samps,GAPs))
