module HMM_MSDDP
using MSDDP, LHS, AR, FFM
using PyCall, Logging, Distributions
@pyimport numpy as np
@pyimport hmmlearn.hmm as hl_hmm

#HMM_MSDDP
export train_hmm, score, predict, inithmm, inithmm_z, inithmm_ar, inithmm_sim, inithmm_ffm
#MSDDP
export MKData, MAAParameters, SDDPParameters, MSDDPModel
export setmarkov!, setnassets!, setγ!, setinistate!, settranscost!
export sddp, simulate, simulatesw, simulate_stateprob, simulatestates, readHMMPara, simulate_percport


function train_hmm(data::Array{Float64,1}, n_states::Int64, lst::Array{Int64,1},
			μ::Array{Float64,1}, σ::Array{Float64,1}; cov_type="full",init_p="")
	μ_ = Array(Float64,n_states,1)
	μ_[:,:] = μ[1:n_states,:]
	σ_ = Array(Float64,n_states,1,1)
	σ_[:,:,:] = σ[1:n_states,:,:]
	debug("Before")
	debug("μ_ ",μ_)
	debug("σ_ ", σ_)
	model = hl_hmm.GaussianHMM(n_components=n_states, covariance_type=cov_type,means_prior=μ_,covars_prior=σ_,
		init_params=init_p)
  data = ((data')') # Has to be Array{Float64,2}

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
	debug("After")
 	debug("μ ", model[:means_])
 	debug("σ ", model[:covars_])
	return model
end

function train_hmm{N}(data::Array{Float64,N}, n_states::Int64; cov_type="full",init_p="stmc")
  if N > 1
    model = hl_hmm.GaussianHMM(n_components=n_states, covariance_type=cov_type,init_params=init_p)
  else
    model = hl_hmm.GaussianHMM(n_components=n_states)
  end
	model[:fit]((data')') # Has to be Array{Float64,2}
	debug("After")
 	debug("μ ", model[:means_])
 	debug("σ ", model[:covars_])
  return model
end

# Return the loglikelihood of the data
function score(model, data::Array{Float64,1})
  model[:score]((data')') # Has to be Array{Float64,2}
end

function score(model, data::Array{Float64,2})
  model[:score](data)
end

# Return the loglikelihood of the data
function score(model, data::Array{Float64,1}, lst::Array{Int64,1})
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

function inithmm(m::MSDDPModel, ret::Array{Float64,2})
  nperiods=size(ret,1)
  samples =1
  return inithmm(m, ret, nperiods, samples)
end

function inithmm_z(m::MSDDPModel, ret::Array{Float64,2}, nperiods::Int64, samples::Int64; pini_cond=true)
  setnassets!(m, nassets(m) +1)
  mk, model = inithmm(m, ret, nperiods, samples, pini_cond=pini_cond)
  setnassets!(m, nassets(m) -1)
  #Remove z state
  mk.ret = mk.ret[1:nassets(m),:,:]
  return mk, model
end
## Uses HMM and LHS to populate MKData for MSDDP
function inithmm(m::MSDDPModel, ret::Array{Float64,2}, nperiods::Int64, samples::Int64; pini_cond=false)
	np.random[:seed](rand(0:4294967295))
  ## Train HMM with data
  lst = fill(nperiods, samples)
  model = train_hmm(ret, nstates(m), lst)

  # Use conditional probability or unconditional probability
	k_ini = (model[:predict](ret[1:nperiods,:]) .+1)[end] # conditional probability
	if !pini_cond
		# The high initial probabilities as the first state
		prob_ini = model[:startprob_] # unconditional probability
		max_prob, k_ini = findmax(prob_ini)
	end


  # Transition matrix (K_t x K_(t+1))
  P_K = model[:transmat_]

  # Conditional probabilities of each state for each scenario p(S|K)
  p_s = ones(nscen(m), nstates(m))*1.0/nscen(m)

  ## Use HMM for each state in LHS
  r = zeros(nstages(m), nassets(m), nstates(m), nscen(m))
  for k = 1:nstates(m)
    μ = reshape(model[:means_][k,:],nassets(m))
    Σ = reshape(model[:covars_][k,:,:], nassets(m), nassets(m))
		debug("μ= ", μ)
		for t = 1:nstages(m)
    	r[t,:,k,:] = lhsnorm(μ, Σ, nscen(m), rando=false)'
		end
  end
  r = exp(r)-1

  dM = MKData(k_ini, P_K, p_s, r)
  return dM, model
end

function inithmm_ar(z::Array{Float64,2}, dF::ARData, m::MSDDPModel, nperiods::Int64, samples::Int64, μ, σ)
	np.random[:seed](rand(0:4294967295))

  lst = fill(nperiods, samples)
  model = train_hmm(reshape(z, (nperiods)*samples), nstates(m), lst, μ, σ)

  # Use z_0 =0
	k_ini = (model[:predict](dF.a_z[1]) .+1)[1] # conditional probability

  # Transition matrix (K_t x K_(t+1))
  P_K = model[:transmat_]

  # Conditional probabilities of each state for each scenario p(S|K)
  p_s = ones(nscen(m), nstates(m))*1.0/nscen(m)

  ## Use HMM for each state in LHS
  r = zeros(nstages(m),nassets(m), nstates(m), nscen(m))
  for k = 1:nstates(m)
		μ = model[:means_][k,1]
    Σ = model[:covars_][k,1]
    z_tp1 = lhsnorm(μ, Σ, nscen(m), rando=false)'
		for t = 1:nstages(m)
			ϵ = lhsnorm(zeros(nassets(m)+1), dF.Σ, nscen(m), rando=true)'
			for s = 1:nscen(m)
				z_t = (z_tp1[s] - dF.a_z[1] - ϵ[nassets(m)+1,s])/dF.b_z[1]
				ρ = dF.a_r + dF.b_r*z_t + ϵ[1:nassets(m),s]
				# Transform ρ = ln(1+r) in return (r)
				r[t,:,k,s] = exp(ρ)-1
				# Discount risk free rate
				r[t,:,k,s] -= dF.r_f
			end
		end
  end
  dM = MKData(k_ini, P_K, p_s, r)
  return dM, model
end

function inithmm_ffm(ff::Array{Float64,2}, dSI::FFMData, m::MSDDPModel)
	np.random[:seed](rand(0:4294967295))

  model = train_hmm(ff, nstates(m))

	# Use conditional probability
	k_ini = (model[:predict](ff) .+1)[end] # conditional probability

  # Transition matrix (K_t x K_(t+1))
  P_K = model[:transmat_]

  # Conditional probabilities of each state for each scenario p(S|K)
  p_s = ones(nscen(m), nstates(m))*1.0/nscen(m)

  ## Use HMM for each state in LHS
  r = zeros(nstages(m), nassets(m), nstates(m), nscen(m))
	samp_ϵ = Array(Float64,nassets(m),nscen(m))
  for k = 1:nstates(m)
		μ = model[:means_][k,:]
    Σ = model[:covars_][k,:,:]
    z = lhsnorm(μ, Σ, nscen(m), rando=false)'
		for t = 1:nstages(m)
			for i = 1:nassets(m)
				samp_ϵ[i,:] = lhsnorm(dSI.μ[i], dSI.σ[i], nscen(m), rando=true)
			end
			for s = 1:nscen(m)
				ρ = dSI.α + (dSI.β'*z[:,s]) + vec(samp_ϵ[:,s])

				# Transform ρ = ln(1+r) in return (r)
				r[t,:,k,s] = exp(ρ)-1
			end
		end
  end
  dM = MKData(k_ini, P_K, p_s, r)
  return dM, model
end

end #HMM_MSDDP
