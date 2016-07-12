module HMM_MSDDP

using MSDDP, LHS, AR
using PyCall, Logging, Distributions
@pyimport numpy as np
@pyimport hmmlearn.hmm as hl_hmm

#HMM_MSDDP
export train_hmm, score, predict, inithmm, inithmm_z, inithmm_ar
#MSDDP
export MKData, MSDDPData
export sddp, simulate, simulatesw, simulate_stateprob, simulatestates, readHMMPara, simulate_percport


function train_hmm(data::Array{Float64,1}, n_states::Int64, lst::Array{Int64,1},
			μ::Array{Float64,1}, σ::Array{Float64,1}; cov_type="full",init_p="")
	μ_ = μ[1:n_states,:]
	σ_ = σ[1:n_states,:,:]
	debug("Before")
	debug("μ_ ",μ_)
	debug("σ_ ", σ_)
	model = hl_hmm.GaussianHMM(n_components=n_states, covariance_type=cov_type,means_prior=μ_,covars_prior=σ_,
		init_params=init_p)
  data = (data')' # Has to be Array{Float64,2}

  model[:fit](data,lst)
	debug("After")
	debug("μ ", model[:means_])
	debug("σ ", model[:covars_])
	return model
end

function train_hmm{N}(data::Array{Float64,N}, n_states::Int64, lst::Array{Int64,1}; cov_type="full",init_p="stmc")
	model = hl_hmm.GaussianHMM(n_components=n_states, covariance_type=cov_type,init_params=init_p)
  if N == 1
    data = (data')' # Has to be Array{Float64,2}
   end
   model[:fit](data,lst)
	return model
end

function train_hmm{N}(data::Array{Float64,N}, n_states::Int64; cov_type="full",init_p="stmc")
  if N > 1
    model = hl_hmm.GaussianHMM(n_components=n_states, covariance_type=cov_type,init_params=init_p)
  else
    model = hl_hmm.GaussianHMM(n_components=n_states)
  end
	model[:fit]((data')') # Has to be Array{Float64,2}
  return model
end

# Return the loglikelihood of the data
function score(model, data)
  model[:score]((data')') # Has to be Array{Float64,2}
end

# Return the loglikelihood of the data
function score(model, data, lst::Array{Int64,1})
  model[:score]((data')',lst) # Has to be Array{Float64,2}
end

function predict(model,data::Array{Float64,1})
	predict(model,(data')')
end
function predict(model,data::Array{Float64,2})
	samples = size(data,1)
	states = Array(Int64,samples)
	for i = 1:samples
		states[i] = (model[:predict](data[1:i,:]) .+1)[end]
	end
	return states
end

function inithmm(ret::Array{Float64,2}, dH::MSDDPData)
  T_l=size(ret,1)
  Sc=1
  return inithmm(ret, dH, T_l, Sc)
end

function inithmm_z(ret::Array{Float64,2}, dH::MSDDPData, T_l::Int64, Sc::Int64; pini_cond=true)
  dH.N += 1
  dM, model = inithmm(ret, dH, T_l, Sc, pini_cond=pini_cond)
  dH.N -= 1
  #Remove z state
  dM.r = dM.r[1:dH.N,:,:]
  return dM, model
end
## Uses HMM and LHS to populate MKData for MSDDP
function inithmm(ret::Array{Float64,2}, dH::MSDDPData, T_l::Int64, Sc::Int64; pini_cond=false)
	np.random[:seed](rand(0:4294967295))
  ## Train HMM with data
  lst = fill(T_l, Sc)
  model = train_hmm(ret, dH.K, lst)

  # Use conditional probability or unconditional probability
	k_ini = (model[:predict](ret[1:T_l,:]) .+1)[end] # conditional probability
	if !pini_cond
		# The high initial probabilities as the first state
		prob_ini = model[:startprob_] # unconditional probability
		max_prob, k_ini = findmax(prob_ini)
	end


  # Transition matrix (K_t x K_(t+1))
  P_K = model[:transmat_]

  # Conditional probabilities of each state for each scenario p(S|K)
  p_s = ones(dH.S, dH.K)*1.0/dH.S

  ## Use HMM for each state in LHS
  r = zeros(dH.N, dH.K, dH.S)
  for k = 1:dH.K
    μ = reshape(model[:means_][k,:],dH.N)
    Σ = reshape(model[:covars_][k,:,:], dH.N, dH.N)
		debug("μ= ", μ)
    r[:,k,:] = lhsnorm(μ, Σ, dH.S, rando=false)'
  end
  r = exp(r)-1

  dM = MKData( r, p_s, k_ini, P_K )
  return dM, model
end

function inithmm_ar(z::Array{Float64,2}, dF::ARData, dH::MSDDPData, T_l::Int64, Sc::Int64, μ, σ)
	np.random[:seed](rand(0:4294967295))

  lst = fill(T_l, Sc)
  model = train_hmm(reshape(z, (T_l)*Sc), dH.K, lst, μ, σ)

  # Use z_0 =0
	k_ini = (model[:predict](dF.a_z[1]) .+1)[1] # conditional probability

  # Transition matrix (K_t x K_(t+1))
  P_K = model[:transmat_]

  # Conditional probabilities of each state for each scenario p(S|K)
  p_s = ones(dH.S, dH.K)*1.0/dH.S

  ## Use HMM for each state in LHS
	norm = MvNormal(dF.Σ)
  r = zeros(dH.N, dH.K, dH.S)
  for k = 1:dH.K
		μ = model[:means_][k,:][:]
    Σ = model[:covars_][k,:]
    z_tp1 = lhsnorm(μ, Σ, dH.S, rando=false)'
		for s = 1:dH.S
      sm = rand(norm)
			e = sm[1:dH.N]
			v = sm[dH.N+1]
			z_t = (z_tp1[s] - dF.a_z[1] - v)/dF.b_z[1]
			ρ = dF.a_r + dF.b_r*z_t + e
			# Transform ρ = ln(1+r) in return (r)
			r[:,k,s] = exp(ρ)-1
			# Discount risk free rate
			r[:,k,s] -= dF.r_f
		end
  end
  dM = MKData( r, p_s, k_ini, P_K )
  return dM, model
end

end #HMM_MSDDP
