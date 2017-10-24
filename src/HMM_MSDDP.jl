module HMM_MSDDP
using MSDDP, LHS, AR, FFM
using PyCall, Logging, Distributions
@pyimport numpy as np
@pyimport hmmlearn.hmm as hl_hmm

#HMM_MSDDP
export train_hmm, score, predict, predictsw, inithmm, inithmm_z, inithmm_ar, inithmm_sim, inithmm_ffm, samplhmm
#MSDDP
export MKData, MAAParameters, SDDPParameters, ModelSizes, MSDDPModel
export solve, simulate, simulatesw, simulate_stateprob, simulatestates, simulate_percport, createmodels!, reset!
export nstages, nassets, nstates, nscen
export setnstages!, setnstates!, setnassets!, setnscen!, setα!, setmarkov!, setγ!, setinistate!, settranscost!
export solve, simulate, simulatesw, simulate_stateprob, simulatestates, simulate_percport, reset!


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

function predict(model, data::Array{Float64,1})
	predict(model,(data')')
end
function predict(model, data::Array{Float64,2})
	samples = size(data,1)
	states = Array(Int64,samples)
	for i = 1:samples
		states[i] = (model[:predict](data[1:i,:]) .+1)[end]
	end
	return states
end

""" Predict function using slide windows """
function predictsw(model, data::Array{Float64,1}, window::Int64)
	predictsw(model,(data')', window)
end

""" Predict function using slide windows """
function predictsw(model, data::Array{Float64,2}, window::Int64)
	samples = size(data,1)
	states = Array(Int64,samples-window+1)
	for i = window:samples
		states[i-window+1] = (model[:predict](data[(i-window+1):i,:]) .+1)[end]
	end
	return states
end

function inithmm(ms::ModelSizes, ret::Array{Float64,2})
  nperiods=size(ret,1)
  samples =1
  return inithmm(ms, ret, nperiods, samples)
end

function inithmm_z(ms::ModelSizes, ret::Array{Float64,2}, nperiods::Int64, samples::Int64; pini_cond=true)
  setnassets!(ms, nassets(ms) +1)
  mk, model = inithmm(ms, ret, nperiods, samples, pini_cond=pini_cond)
  setnassets!(ms, nassets(ms) -1)
  #Remove z state
  mk.ret = mk.ret[1:nassets(ms),:,:]
  return mk, model
end
## Uses HMM and LHS to populate MKData for MSDDP
function inithmm(ms::ModelSizes, ret::Array{Float64,2}, nperiods::Int64, samples::Int64; pini_cond=false)
	np.random[:seed](rand(0:4294967295))
  ## Train HMM with data
  lst = fill(nperiods, samples)
  model = train_hmm(ret, nstates(ms), lst)

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
  p_s = ones(nscen(ms), nstates(ms))*1.0/nscen(ms)

  ## Use HMM for each state in LHS
  r = zeros(nstages(ms), nassets(ms), nstates(ms), nscen(ms))
  for k = 1:nstates(ms)
    μ = reshape(model[:means_][k,:],nassets(ms))
    Σ = reshape(model[:covars_][k,:,:], nassets(ms), nassets(ms))
		debug("μ= ", μ)
		for t = 1:nstages(ms)
    	r[t,:,k,:] = lhsnorm(μ, Σ, nscen(ms), rando=false)'
		end
  end
  r = exp(r)-1

  dM = MKData(k_ini, P_K, p_s, r)
  return dM, model
end

function inithmm_ar(z::Array{Float64,2}, dF::ARData, ms::ModelSizes, nperiods::Int64, samples::Int64, μ, σ)
	np.random[:seed](rand(0:4294967295))

  lst = fill(nperiods, samples)
  model = train_hmm(reshape(z, (nperiods)*samples), nstates(ms), lst, μ, σ)

  # Use z_0 =0
	k_ini = (model[:predict](dF.a_z[1]) .+1)[1] # conditional probability

  # Transition matrix (K_t x K_(t+1))
  P_K = model[:transmat_]

  # Conditional probabilities of each state for each scenario p(S|K)
  p_s = ones(nscen(ms), nstates(ms))*1.0/nscen(ms)

  ## Use HMM for each state in LHS
  r = zeros(nstages(ms),nassets(ms), nstates(ms), nscen(ms))
  for k = 1:nstates(ms)
		μ = model[:means_][k,1]
    Σ = model[:covars_][k,1]
    z_tp1 = lhsnorm(μ, Σ, nscen(ms), rando=false)'
		for t = 1:nstages(ms)
			ϵ = lhsnorm(zeros(nassets(ms)+1), dF.Σ, nscen(ms), rando=true)'
			for s = 1:nscen(ms)
				z_t = (z_tp1[s] - dF.a_z[1] - ϵ[nassets(ms)+1,s])/dF.b_z[1]
				ρ = dF.a_r + dF.b_r*z_t + ϵ[1:nassets(ms),s]
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

function inithmm_ffm(ff::Array{Float64,2}, dSI::FFMData, ms::ModelSizes)
	np.random[:seed](rand(0:4294967295))

  model = train_hmm(ff, nstates(ms))

	# Use conditional probability
	k_ini = (model[:predict](ff) .+1)[end] # conditional probability

  # Transition matrix (K_t x K_(t+1))
  P_K = model[:transmat_]

  # Conditional probabilities of each state for each scenario p(S|K)
  p_s = ones(nscen(ms), nstates(ms))*1.0/nscen(ms)

  ## Use HMM for each state in LHS
  r = zeros(nstages(ms), nassets(ms), nstates(ms), nscen(ms))
	samp_ϵ = Array(Float64,nassets(ms),nscen(ms))
  for k = 1:nstates(ms)
		μ = model[:means_][k,:]
    Σ = model[:covars_][k,:,:]
    z = lhsnorm(μ, Σ, nscen(ms), rando=false)'
		for t = 1:nstages(ms)
			for i = 1:nassets(ms)
				samp_ϵ[i,:] = lhsnorm(dSI.μ[i], dSI.σ[i], nscen(ms), rando=true)
			end
			for s = 1:nscen(ms)
				ρ = dSI.α + (dSI.β'*z[:,s]) + vec(samp_ϵ[:,s])

				# Transform ρ = ln(1+r) in return (r)
				r[t,:,k,s] = exp(ρ)-1
			end
		end
  end
  dM = MKData(k_ini, P_K, p_s, r)
  return dM, model
end

function samplhmm(dSI::FFMData, ms::ModelSizes, model, n_scen::Int)
    np.random[:seed](rand(0:4294967295))

    r = zeros(nassets(ms), nstates(ms), n_scen)
    samp_ϵ = Array(Float64,nassets(ms), n_scen)
    for k = 1:nstates(ms)
        μ = model[:means_][k,:]
        Σ = model[:covars_][k,:,:]
        z = lhsnorm(μ, Σ, n_scen, rando=false)'
        for i = 1:nassets(ms)
            samp_ϵ[i,:] = lhsnorm(dSI.μ[i], dSI.σ[i], n_scen, rando=true)
        end
        for s = 1:n_scen
            ρ = dSI.α + (dSI.β'*z[:,s]) + vec(samp_ϵ[:,s])
            # Transform ρ = ln(1+r) in return (r)
            r[:,k,s] = exp(ρ)-1
        end

    end
	return r
end
end #HMM_MSDDP
