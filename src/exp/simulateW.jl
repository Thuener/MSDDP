include("parameters.jl")
output_dir = "../../output2/"

NS = 1000
series = zeros(Float64,dH.N,dH.T,NS)
states = zeros(Int64,dH.T,NS)
K_forward = Array(Int64,dH.T)
r_forward = zeros(dH.N,dH.T)
for s = 1:NS
  simulatestates(dH, dM, K_forward, r_forward)
  series[:,:,s] = r_forward
  states[:,s] = K_forward
end

γs = [0.005,0.02]
cs = [0.02]

file = string(output_dir,file_name,"_ret.csv")
# For each risk level (γ)
for i_γ = 1:length(γs)
  dH.γ = γs[i_γ]
  # For each transactional cost (c)
  for i_c = 1:length(cs)
    dH.c = cs[i_c]
    ret,rets = runMSDDP_TD_TC_IN(deepcopy(dH), dM, series, states, dH.T)
    γ_srt = string(dH.γ)[3:end]
    c_srt = string(dH.c)[3:end]

    writecsv(string(output_dir,file_name,"_rets_g$(γ_srt)_c$(c_srt).csv"),rets)
  end
end

run(`/home/tas/woofy.sh 62491240 "simulateW"`)
