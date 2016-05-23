module HMM_MSDDP

using HMM
using MSDDP
using LHS

export inithmm
#MSDDP
export HMMData, MSDDPData
export SDDP, simulate, SimulateStates, readHMMPara, simulatePercPort

function inithmm(ret::Array{Float64,2}, dH::MSDDPData)
  T_l=size(ret,1)
  Sc=1
  return inithmm(ret, dH, T_l, Sc)
end
## Uses HMM and LHS to populate HMMData for MSDDP
function inithmm(ret::Array{Float64,2}, dH::MSDDPData, T_l::Int64, Sc::Int64)
  ## Train HMM with data
  lst = fill(T_l, Sc)
  model = train_hmm(ret,dH.K,lst)

  # Use the high initial probabilities as the first state
  max_prob, k_ini = findmax(model[:startprob_])

  # Transition matrix (K_t x K_(t+1))
  P_K = model[:transmat_]

  # Conditional probabilities of each state for each scenario p(S|K)
  p_s = ones(dH.S, dH.K)*1.0/dH.S

  ## Use HMM for each state in LHS
  r = zeros(dH.N, dH.K, dH.S)
  for k = 1:dH.K
    μ = reshape(model[:means_][k,:],dH.N)
    Σ = reshape(model[:covars_][k,:,:], dH.N, dH.N)
    r[:,k,:] = lhsnorm(μ, Σ, dH.S, rando=false)'
  end
  r = exp(r)-1

  dM = HMMData( r, p_s, k_ini, P_K )
  return dM
end

end #HMM_MSDDP
